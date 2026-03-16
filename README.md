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
Traceback (most recent call last):
  File "/home/vipuser/.conda/envs/medsam/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/vipuser/.conda/envs/medsam/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/vipuser/testMedSAM/pilot_phase1/run_phase1.py", line 30, in <module>
    from segment_anything import sam_model_registry
  File "/home/vipuser/testMedSAM/segment_anything/__init__.py", line 15, in <module>
    from .predictor import SamPredictor
  File "/home/vipuser/testMedSAM/segment_anything/predictor.py", line 15, in <module>
    from .utils.transforms import ResizeLongestSide
  File "/home/vipuser/testMedSAM/segment_anything/utils/transforms.py", line 11, in <module>
    from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
  File "/home/vipuser/.conda/envs/medsam/lib/python3.10/site-packages/torchvision/__init__.py", line 10, in <module>
    from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip
  File "/home/vipuser/.conda/envs/medsam/lib/python3.10/site-packages/torchvision/_meta_registrations.py", line 164, in <module>
    def meta_nms(dets, scores, iou_threshold):
  File "/home/vipuser/.conda/envs/medsam/lib/python3.10/site-packages/torch/library.py", line 1073, in register
    use_lib._register_fake(
  File "/home/vipuser/.conda/envs/medsam/lib/python3.10/site-packages/torch/library.py", line 203, in _register_fake
    handle = entry.fake_impl.register(
  File "/home/vipuser/.conda/envs/medsam/lib/python3.10/site-packages/torch/_library/fake_impl.py", line 50, in register
    if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
RuntimeError: operator torchvision::nms does not exist
--data_root：未找到命令
--checkpoint：未找到命令
--device：未找到命令
--n_calibration：未找到命令
--output_dir：未找到命令
