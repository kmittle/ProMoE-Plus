routing_logits = self.mos_router(
    prompt_embedding=prompt_embedding,  # [L_txt, C_prompt]
    time_embedding=e0_router,            # [L_vis, 6*C]
    z_t=vis,                             # [L_vis, C]
    seqlens_txt=seqlens_txt,
    seqlens_vis=seqlens_vis,
    freqs_txt=freqs_txt,
    freqs_vis=freqs_only_vis,
    training=self.training
)  # [L_txt, m, n] ✅ 一次性生成完整的路由矩阵

# ✅ 在外层进行 softmax，得到权重矩阵
routing_weights_list = []
for sample_idx, routing_logits_i in enumerate(routing_logits.split(seqlens_txt)):
    # routing_logits_i: [L_txt_i, m, n]
    # ✅ 沿 m 维度做 softmax，使得每个 token 对每个 gen_layer，m 个 understanding layers 的权重和为 1
    routing_weights_i = torch.softmax(routing_logits_i, dim=1)  # [L_txt_i, m, n]
    routing_weights_list.append(routing_weights_i)

routing_weights = torch.cat(routing_weights_list, dim=0)  # [L_txt, m, n]


class MoSRouter(nn.Module):
    """
    MoS Router - 正确版本
    
    输入：
    - c: 原始prompt embedding (输入到U的)
    - t: timestep
    - z_t: noised image latent
    
    输出：
    - W: [L_txt, m, n] logit matrix
          其中L_txt是prompt长度，m是understanding深度，n是generation深度
    """
    
    def __init__(
        self,
        prompt_dim: int,      # c的维度（原始prompt embedding的维度）
        time_dim: int,         # timestep embedding的维度
        gen_dim: int,          # generation latent的维度
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        und_num_layers: int,   # m
        gen_num_layers: int,   # n
        qk_norm = True,
        eps = 1e-6,
        router_hidden_dim: int = 256,
        num_router_blocks: int = 2,
        k: int = 2,
        epsilon: float = 0.05
    ):
        super().__init__()
        
        self.und_num_layers = und_num_layers
        self.gen_num_layers = gen_num_layers
        self.k = k
        self.epsilon = epsilon
        
        # ===== 输入投影层 =====
        # prompt embedding (c) 投影
        self.prompt_norm = RMSNorm(prompt_dim, eps)
        self.prompt_proj = nn.Linear(prompt_dim, router_hidden_dim, bias=False)
        
        # timestep投影
        self.time_norm = RMSNorm(time_dim, eps)
        self.time_proj = nn.Linear(time_dim, router_hidden_dim, bias=False)
        
        # latent投影
        self.latent_norm = RMSNorm(gen_dim, eps)
        self.latent_proj = nn.Linear(gen_dim, router_hidden_dim, bias=False)
        
        # ===== Router Transformer blocks =====
        self.router_blocks = nn.ModuleList([
            TransformerBlock(router_hidden_dim, num_heads, head_dim, mlp_dim, qk_norm, eps)
            for _ in range(num_router_blocks)
        ])
        
        # ===== 输出投影：生成m×n的logit matrix =====
        # 每个token预测一个m×n的矩阵
        self.routing_head = nn.Linear(router_hidden_dim, und_num_layers * gen_num_layers, bias=False)

        # self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        prompt_embedding: torch.Tensor,  # [L_txt, C_prompt]
        time_embedding: torch.Tensor,     # [L_vis, time_dim]
        z_t: torch.Tensor,                # [L_vis, C_gen]
        seqlens_txt: List[int],
        seqlens_vis: List[int],
        freqs_txt,                         # [2, L_txt, D] 只有txt，按sample组织
        freqs_vis,                    # [2, L_vis, D] 只有vis，按sample组织
        training: bool = True
    ) -> torch.Tensor:
        
        prompt_splits = prompt_embedding.split(seqlens_txt)
        time_splits = time_embedding.split(seqlens_vis)
        z_t_splits = z_t.split(seqlens_vis)
        
        # ✅ 按sample拆分frequencies
        freqs_txt_splits = freqs_txt.split(seqlens_txt, dim=1)  # list of [2, L_txt_i, D]
        freqs_vis_splits = freqs_vis.split(seqlens_vis, dim=1)  # list of [2, L_vis_i, D]
        
        all_routing_logits = []
        
        for sample_idx, (prompt_i, time_i, z_t_i, freqs_txt_i, freqs_vis_i) in enumerate(
            zip(prompt_splits, time_splits, z_t_splits, freqs_txt_splits, freqs_vis_splits)
        ):  
            # assert not torch.isnan(prompt_i).any(), f"prompt_i has NaN at sample {sample_idx}"
            # assert not torch.isnan(time_i).any(), f"time_i has NaN at sample {sample_idx}"
            # assert not torch.isnan(z_t_i).any(), f"z_t_i has NaN at sample {sample_idx}"
            # ===== 步骤1：投影 =====
            prompt_norm = self.prompt_norm(prompt_i)
            # assert not torch.isnan(prompt_norm).any(), f"prompt_norm has NaN"
            prompt_proj = self.prompt_proj(prompt_norm)
            # assert not torch.isnan(prompt_proj).any(), f"prompt_proj has NaN"
            
            time_norm = self.time_norm(time_i)
            # assert not torch.isnan(time_norm).any(), f"time_norm has NaN"
            time_proj = self.time_proj(time_norm)
            # assert not torch.isnan(time_proj).any(), f"time_proj has NaN"
            
            latent_norm = self.latent_norm(z_t_i)
            # assert not torch.isnan(latent_norm).any(), f"latent_norm has NaN"
            latent_proj = self.latent_proj(latent_norm)
            # assert not torch.isnan(latent_proj).any(), f"latent_proj has NaN"
            
            # ===== 步骤2：拼接 =====
            combined_input = torch.cat([prompt_proj, time_proj, latent_proj], dim=0)
            # [L_txt_i + 2*L_vis_i, router_dim]
            
            # ✅ 拼接frequencies：txt + vis
            freqs_combined_i = torch.cat([freqs_txt_i, freqs_vis_i, freqs_vis_i], dim=1)
            # [2, L_txt_i + 2*L_vis_i, D]
            
            combined_seqlen_i = seqlens_txt[sample_idx] + 2 * seqlens_vis[sample_idx]
            
            # ===== 步骤3：通过Router Transformer =====
            hidden = combined_input
            for block in self.router_blocks:
                hidden = block(hidden, seqlens=[combined_seqlen_i], freqs=freqs_combined_i)
            
            # ===== 步骤4：生成routing logits =====
            prompt_hidden = hidden[:prompt_i.shape[0]]
            logits_flat = self.routing_head(prompt_hidden)
            routing_logits_i = logits_flat.reshape(prompt_i.shape[0], self.und_num_layers, self.gen_num_layers)
            
            all_routing_logits.append(routing_logits_i)
        
        routing_logits = torch.cat(all_routing_logits, dim=0)
        return routing_logits


self.mos_router = MoSRouter(
    prompt_dim=dim,
    time_dim=dim * 6,
    gen_dim=dim,
    num_heads=self.num_heads,
    head_dim=self.head_dim,
    mlp_dim=self.mlp_dim,
    und_num_layers=len(self.blocks),  # m
    gen_num_layers=len(self.blocks),  # n
    router_hidden_dim=dim,
    num_router_blocks=2,
    k=2,
    epsilon=0.05
)
