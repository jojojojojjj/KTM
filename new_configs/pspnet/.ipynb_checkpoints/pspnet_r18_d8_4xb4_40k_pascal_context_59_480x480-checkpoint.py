from mmengine.config import read_base
with read_base():
    from .pspnet_r50_d8_4xb4_40k_pascal_context_59_480x480 import * # noqa

model.update(dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64)))

work_dir='autodl-tmp/lhf/HRDKD-master/work_dirs_pascal_context/pspnet_r18_d8_4xb4_40k_pascal_context_59_480x480'