from mmengine.config import read_base
with read_base():
    from .._base_.models.deeplabv3plus_r50_d8 import * # noqa 
    from .._base_.datasets.cityscapes import * # noqa
    from .._base_.default_runtime import * # noqa
    from .._base_.schedules.schedule_40k import * # noqa

crop_size = (512, 1024)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(data_preprocessor=data_preprocessor))
work_dir='autodl-tmp/lhf/HRDKD-master/work_dirs/deeplabv3plus_r50_d8_4xb2_40k_cityscapes_512x1024'