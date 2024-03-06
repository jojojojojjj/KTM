from mmengine.config import read_base
from seg.models.losses.rktm import RKTMLoss
with read_base():
    from ..deeplabv3p.shufflenet_v2_d8_deeplabv3plus_4xb2_40k_cityscapes_512x1024 import * # noqa

model.update(dict(
    decode_head=dict(
        loss_decode=[
            dict(
            type=CrossEntropyLoss,use_sigmoid=False, loss_weight=1.0),
                        dict(
            type=RKTMLoss, kd='dist',pooling=False,loss_weight=5.0),
        ]
    ),
    auxiliary_head=dict(
    loss_decode=[
            dict(
            type=CrossEntropyLoss,use_sigmoid=False, loss_weight=0.4),
                        dict(
            type=RKTMLoss, kd='dist',pooling=False,loss_weight=2),
        ]
    )))

work_dir='yourpath'