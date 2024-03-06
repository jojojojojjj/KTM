from mmengine.config import read_base
from seg.models.losses.rktm import RKTMLoss
with read_base():
    from ..pspnet.pspnet_r18_d8_4xb4_40k_pascal_context_59_480x480 import * # noqa

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