# Run-time commands — 2026_06_21

> 启动命令在当前 tmux 会话的新窗口中运行，例如：
> `tmux new-window -t "$(tmux display-message -p '#S')" -n <name> '<启动命令>'`
> git 分支一列反映生成此表时的当前分支（`repa`）；若各实验目标分支不同请自行核对。

实验描述 | git分支 | 启动命令 | 输出位置
--- | --- | --- | ---
Slot 1.1 · GPU 0-3 · ProMoE-B EC-BC proto_t (direct) | repa | `bash scripts/_run_times/2026_06_21/1.1-B_ec_bc_proto_t_direct.sh` | `outputs/ProMoE_EC_BC_B_proto_t/004_ProMoE_B_EC_BC_proto_t_direct/`
Slot 1.2 · GPU 4-7 · ProMoE-B TC proto_t (direct) | repa | `bash scripts/_run_times/2026_06_21/1.2-B_tc_proto_t_direct.sh` | `outputs/ProMoE_TC_B_proto_t/004_ProMoE_B_proto_t_direct/`
Slot 2.1 · GPU 0-3 · ProMoE-B EC-BC proto_t (residual) | repa | `bash scripts/_run_times/2026_06_21/2.1-B_ec_bc_proto_t_residual.sh` | `outputs/ProMoE_EC_BC_B_proto_t/004_ProMoE_B_EC_BC_proto_t_residual/`
Slot 2.2 · GPU 4-7 · ProMoE-B TC proto_t (residual) | repa | `bash scripts/_run_times/2026_06_21/2.2-B_tc_proto_t_residual.sh` | `outputs/ProMoE_TC_B_proto_t/004_ProMoE_B_proto_t_residual/`
