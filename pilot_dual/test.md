cd /home/zhangcenwei/testSAMpruning

/home/zhangcenwei/miniconda3/envs/medsam/bin/python -m pilot_dual.run_dual \
    --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
    --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \
    --data_root   assert/CVC-ColonDB \
    --device      cuda:0 \
    --n_cal       128 \
    --batch_size  1 \
    --sparsities  0.3 0.5 0.7 \
    --alpha_values 0.0 0.2 0.5 0.8 1.0 \
    --output_dir  results/pilot_dual



cd /home/zhangcenwei/testSAMpruning

/home/zhangcenwei/miniconda3/envs/medsam/bin/python -m pilot_dual.run_dual \
    --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
    --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \
    --data_root   assert/CVC-ColonDB \
    --device      cuda:1 \
    --output_dir  results/pilot_dual


# cascade: head pruning (Phase 1) then MLP pruning (Phase 2)
cd /home/zhangcenwei/testSAMpruning

/home/zhangcenwei/miniconda3/envs/medsam/bin/python -m pilot_dual.run_cascade \
    --medsam_ckpt  work_dir/MedSAM/medsam_vit_b.pth \
    --sam_ckpt     work_dir/SAM/sam_vit_b_01ec64.pth \
    --data_root    assert/CVC-ColonDB \
    --device       cuda:1 \
    --output_dir   results/pilot_cascade
