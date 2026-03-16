## 1.测试数据集

- CVC-ColonDB: https://www.kaggle.com/datasets/longvil/cvc-colondb

## 2. 需要手动创建输出目录，原始MedSAM权重需要放在work_dir

- 输出目录: work_dir
- 原始MedSAM权重放在work_dir/MedSAM/medsam_vit_b: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN 


## 3. 测试运行
python -m pilot_phase1.run_phase1 \
    --data_root assert/CVC-ColonDB \
    --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --device cuda:0 \
    --n_calibration 128 \
    --output_dir results/phase1
