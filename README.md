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

## problems
[Step 4] Computing head importance scores...
Traceback (most recent call last):
  File "/home/vipuser/.conda/envs/medsam/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/vipuser/.conda/envs/medsam/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/vipuser/testMedSAM/pilot_phase1/run_phase1.py", line 328, in <module>
    main()
  File "/home/vipuser/testMedSAM/pilot_phase1/run_phase1.py", line 131, in main
    head_importance, per_sample_proj = compute_head_gradient_projections_fast(
  File "/home/vipuser/testMedSAM/pilot_phase1/head_pruning.py", line 208, in compute_head_gradient_projections_fast
    seg_loss_fn = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
NameError: name 'monai' is not defined
--data_root：未找到命令
--checkpoint：未找到命令
--device：未找到命令
--n_calibration：未找到命令
--output_dir：未找到命令
