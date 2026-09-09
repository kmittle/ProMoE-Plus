# 总体目标: 利用条件专家的路由信息辅助最终对齐

## 之前的实验发现

如你在"models"目录下所见，我们之前对repa的对齐位置进行了大量的消融，具体来说，我们发现仅仅对conditional专家对齐效果不好、对conditional专家+unconditional专家对齐效果也不好，在transformer block输出的地方对齐效果最好（naive repa），仅仅对shared专家对齐效果还行，比在block之后对齐效果相当（仅仅略差于naive repa）。

关于上面的实验的具体实现细节，你都可以在本项目的模型文件、配置文件、sh实验脚本中找到。如果你对我描述的任何信息有疑惑，请到本项目中寻找信息，或者主动与我沟通，这是为了严格保证实现细节与接下来的实验意图一致。（此处理解请回复“我会做到结合本项目信息理解意图”）

## 我们的新的想法

这说明naive repa（models/models_ProMoE_TC_repa.py）的对齐位置可能确实是最好的对齐位置，但是同时，我们坚信conditional专家的路由信息是有益于repa对齐的。但是我们不能直接在conditional专家的输出位置直接对齐，如上所述，这并不能带来很好的效果。

为此，我们拟进行一系列消融实验。
总体来说，我们的对齐位置还是保持与naive repa相同（在transformer block的输出之后）；不同的是，我们不再让DiT的每个token与DINO侧对应的那一个token对齐。具体来说，我们认为对于conditional专家来说，被分到同一个conditional专家且属于同一张图的token是高度关联的，这些token应该互相与它们对应的DINO token对齐（此处是否理解？如果此处理解请回复：“总体思想已理解”）

## 技术实现细节部分

尽管我们认为被分到同一个专家且属于同一张的图的DiT token应该与它们对应的DINO token互相对齐，直白地说：假设DiT的token a、b、c属于同一张图且被分到了同一个conditional专家（以下提到token a、b、c都默认这个设定），那么DiT的tokena、b、c都需要与a、b、c对应的DINO token对齐。但是我们同时认为，他们的对齐权重不应该是一样的。具体来说，我们认为这里有三种策略：

1. 使用一个transformer模块，计算整张图的attention map（注意必须是某张图内的token计算attention map），针对token a，使用token a与token b、token c的attention map的权重作为token a与各个对应的DINO token对齐的动态权重。这里还有两个值得注意的地方：
    - 在patchify后，输入DiT transformer block前，用一个2层的transformer block先对token做处理，再使用处理后的token做qkv投影，再计算scale dot product来得到attention map，后续block对齐时，复用这个attention map。
    - 对于需要进行repa对齐的block，对该block输出的token使用1或2层（经典repa模式仅对齐1个block的情况下就用2层可学习的transformer block预处理；当需要对齐的block超过1个时，仅使用1层可学习的transformer block预处理，这种情况可能在MoS对齐框架中出现）可学习的transformer block预处理，再经过qkv投影和scale dot product得到block-wise的attention map，使用这个attention map作为权重。
2. 与case 1是类似的，区别在于，attention map并不是对全图算的，而是仅限于属于同一张图且分到了同一个conditional专家的token。因此，在这个设定下，与case 1的区别是没有patchify后、输入DiT前的计算attention map这种情况了，因为必须在moe block路由后才知道同一张图的哪些token被分到了同一个专家。这里我们进一步说明为什么一直强调这些token必须是同一张图的：因为不同的图的token可能加噪程度不同，其实是不适合直接做attention计算的。
3. 在路由阶段，被分到同一个专家的token都会与prototype计算相似度，这个相似度其实也是很好的repa对齐权重。举例来说，我们的想法是：对于token a，其与自身DINO token的对齐权重固定为1，与token b对应的DINO token的对齐权重为token a路由时的余弦相似度乘上token b路由时的余弦相似度。

综上所述，如何确定属于同一张图且分到同一个专家的token相互对齐的权重，总共有4种方案。

## 实验规划

我们计划在naive repa（models/models_ProMoE_TC_repa.py）单层对齐和MoS多层对齐（models/models_ProMoE_TC_repa_MoS_naive_choice.py）上都实现上述想法，所以一共应该是8个实验。
