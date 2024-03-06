from mmengine.config import read_base
from seg.models.losses.nrktm import NRKTMLoss
with read_base():
    from ..deeplabv3p.shufflenet_v2_d8_deeplabv3plus_4xb2_40k_pascal_context_480x480 import * # noqa

model.update(dict(
    decode_head=dict(
        loss_decode=[
            dict(
            type=CrossEntropyLoss,use_sigmoid=False, loss_weight=1.0),
                        dict(
            type=NRKTMLoss, kd='cwd', tau=4,p_max=80,p_min=-3,pooling=False,loss_weight=1.0),
        ]
    ),
    auxiliary_head=dict(
    loss_decode=[
            dict(
            type=CrossEntropyLoss,use_sigmoid=False, loss_weight=0.4),
                        dict(
            type=NRKTMLoss, kd='cwd', tau=4,p_max=80,p_min=-3,pooling=False,loss_weight=0.4),
        ]
    )))

work_dir='yourpath'