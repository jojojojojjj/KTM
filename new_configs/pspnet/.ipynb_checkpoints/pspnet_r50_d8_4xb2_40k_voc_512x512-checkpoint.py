from mmengine.config import read_base
with read_base():
    from .._base_.models.pspnet_r50_d8 import * # noqa 
    from .._base_.datasets.pascal_voc12_aug import * # noqa
    from .._base_.default_runtime import * # noqa
    from .._base_.schedules.schedule_40k import * # noqa

crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(num_classes=21)))
work_dir='autodl-tmp/lhf/HRDKD-master/work_dirs_batch6/pspnet_r18_d8_4xb2_40k_voc_512x512'