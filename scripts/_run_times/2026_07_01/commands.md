实验描述 | git分支 | 启动命令 | 输出位置
--- | --- | --- | ---
Slot 1.1 · GPU 0-3 · ProMoE-B LB-Contra reweight β=0.25 | repa | `bash scripts/_run_times/2026_07_01/1.1-B_lbcontra_reweight_b0p25.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_reweight_b0p25/`
Slot 1.2 · GPU 4-7 · ProMoE-B LB-Contra reweight β=0.5 | repa | `bash scripts/_run_times/2026_07_01/1.2-B_lbcontra_reweight_b0p5.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_reweight_b0p5/`
Slot 2.1 · GPU 0-3 · ProMoE-B LB-Contra reweight β=1 | repa | `bash scripts/_run_times/2026_07_01/2.1-B_lbcontra_reweight_b1.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_reweight_b1/`
Slot 2.2 · GPU 4-7 · ProMoE-B LB-Contra reweight β=2 | repa | `bash scripts/_run_times/2026_07_01/2.2-B_lbcontra_reweight_b2.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_reweight_b2/`
Slot 3.1 · GPU 0-3 · ProMoE-B LB-Contra logit-adjust τ=0.5 | repa | `bash scripts/_run_times/2026_07_01/3.1-B_lbcontra_logitadj_t0p5.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_logitadj_t0p5/`
Slot 3.2 · GPU 4-7 · ProMoE-B LB-Contra logit-adjust τ=1 | repa | `bash scripts/_run_times/2026_07_01/3.2-B_lbcontra_logitadj_t1.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_logitadj_t1/`
Slot 4.1 · GPU 0-3 · ProMoE-B LB-Contra logit-adjust τ=2 | repa | `bash scripts/_run_times/2026_07_01/4.1-B_lbcontra_logitadj_t2.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_logitadj_t2/`
Slot 4.2 · GPU 4-7 · ProMoE-B LB-Contra logit-adjust τ=4 | repa | `bash scripts/_run_times/2026_07_01/4.2-B_lbcontra_logitadj_t4.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_logitadj_t4/`
Slot 5.1 · GPU 0-3 · ProMoE-B LB-Contra balance λ=0.001 | repa | `bash scripts/_run_times/2026_07_01/5.1-B_lbcontra_balance_l0p001.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_balance_l0p001/`
Slot 5.2 · GPU 4-7 · ProMoE-B LB-Contra balance λ=0.01 | repa | `bash scripts/_run_times/2026_07_01/5.2-B_lbcontra_balance_l0p01.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_balance_l0p01/`
Slot 6.1 · GPU 0-3 · ProMoE-B LB-Contra balance λ=0.1 | repa | `bash scripts/_run_times/2026_07_01/6.1-B_lbcontra_balance_l0p1.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_balance_l0p1/`
Slot 6.2 · GPU 4-7 · ProMoE-B LB-Contra balance λ=1 | repa | `bash scripts/_run_times/2026_07_01/6.2-B_lbcontra_balance_l1.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_balance_l1/`
Slot 7.1 · GPU 0-3 · ProMoE-B LB-Contra soft-only | repa | `bash scripts/_run_times/2026_07_01/7.1-B_lbcontra_soft_only.sh` | `outputs/ProMoE_TC_B_lbcontra/004_ProMoE_B_lbcontra_soft_only/`
Slot 7.2 · GPU 4-7 · ProMoE-B DAG-Fuse cond-from-shared | repa | `bash scripts/_run_times/2026_07_01/7.2-B_dagfuse_condfromshared.sh` | `outputs/ProMoE_TC_B_dagfuse/004_ProMoE_B_dagfuse_condfromshared/`
Slot 8.1 · GPU 0-3 · ProMoE-B DAG-Fuse shared-from-cond | repa | `bash scripts/_run_times/2026_07_01/8.1-B_dagfuse_sharedfromcond.sh` | `outputs/ProMoE_TC_B_dagfuse/004_ProMoE_B_dagfuse_sharedfromcond/`
Slot 8.2 · GPU 4-7 · ProMoE-B DAG-Fuse bidirectional | repa | `bash scripts/_run_times/2026_07_01/8.2-B_dagfuse_bidirectional.sh` | `outputs/ProMoE_TC_B_dagfuse/004_ProMoE_B_dagfuse_bidirectional/`
Slot 9.1 · GPU 0-3 · ProMoE-B A-Depth q=0.1 | repa | `bash scripts/_run_times/2026_07_01/9.1-B_adepth_q0p1.sh` | `outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p1/`
Slot 9.2 · GPU 4-7 · ProMoE-B A-Depth q=0.2 | repa | `bash scripts/_run_times/2026_07_01/9.2-B_adepth_q0p2.sh` | `outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p2/`
Slot 10.1 · GPU 0-3 · ProMoE-B A-Depth q=0.3 | repa | `bash scripts/_run_times/2026_07_01/10.1-B_adepth_q0p3.sh` | `outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p3/`
Slot 10.2 · GPU 4-7 · ProMoE-B A-Depth q=0.4 | repa | `bash scripts/_run_times/2026_07_01/10.2-B_adepth_q0p4.sh` | `outputs/ProMoE_TC_B_adepth/004_ProMoE_B_adepth_q0p4/`
Slot 11.1 · GPU 0-3 · ProMoE-B Loss-Free u=1e-4 | repa | `bash scripts/_run_times/2026_07_01/11.1-B_lossfree_u1e4.sh` | `outputs/ProMoE_TC_B_lossfree/004_ProMoE_B_lossfree_u1e4/`
Slot 11.2 · GPU 4-7 · ProMoE-B Loss-Free u=1e-3 | repa | `bash scripts/_run_times/2026_07_01/11.2-B_lossfree_u1e3.sh` | `outputs/ProMoE_TC_B_lossfree/004_ProMoE_B_lossfree_u1e3/`
Slot 12.1 · GPU 0-3 · ProMoE-B Loss-Free u=1e-2 | repa | `bash scripts/_run_times/2026_07_01/12.1-B_lossfree_u1e2.sh` | `outputs/ProMoE_TC_B_lossfree/004_ProMoE_B_lossfree_u1e2/`
