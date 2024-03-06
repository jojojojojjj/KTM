from mmengine.config import read_base
with read_base():
    from .deeplabv3plus_r50_d8_4xb2_40k_cityscapes_512x1024 import * # noqa


model.update(dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64)))
work_dir='autodl-tmp/lhf/HRDKD-master/work_dirs_batch6/deeplabv3plus_r18b_d8_4xb2_40k_cityscapes_512x1024'