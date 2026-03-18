## 1.测试数据集

- CVC-ColonDB: https://www.kaggle.com/datasets/longvil/cvc-colondb

## 2. 需要手动创建输出目录，原始MedSAM权重需要放在work_dir

- 输出目录: work_dir
- 原始MedSAM权重放在work_dir/MedSAM/medsam_vit_b: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN 

## 3. Some installation

1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/bowang-lab/MedSAM`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


## 4. 测试运行
### Check 1
python -m pilot_phase1.check1_proxy \
    --data_root assert/CVC-ColonDB \
    --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --device cuda:0 \
    --n_calibration 128 \
    --check_blocks 0 6 11 \
    --output_dir results/phase1_v2

### Check 3(fast)
python -m pilot_phase1.run_phase1 \
    --data_root assert/CVC-ColonDB \
    --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --device cuda:0 \
    --n_calibration 128 \
    --sparsities 0.5 \
    --alpha_values 0.1 0.5 1.0 \
    --check_eps_sensitivity \
    --sinkhorn_iters 100 \
    --output_dir results/phase1_v2

### Check 2
python -m pilot_phase1.run_phase1 \
    --data_root assert/CVC-ColonDB \
    --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --device cuda:1 \
    --n_calibration 128 \
    --sparsities 0.5 \
    --alpha_values 0.1 0.5 1.0 \
    --sinkhorn_iters 100 \
    --output_dir results/phase1_v2

### Final complete
python -m pilot_phase1.run_phase1 \
    --data_root assert/CVC-ColonDB \
    --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --device cuda:2 \
    --n_calibration 128 \
    --sparsities 0.3 0.5 0.7 \
    --alpha_values 0.1 0.5 1.0 \
    --sinkhorn_iters 100 \
    --include_legacy \
    --output_dir results/phase1_v2


## problems
none
