from mmengine.config import read_base
from seg.models.losses.kd_loss_gtst import GTSTLoss
with read_base():
    from ..pspnet.pspnet_r18_d8_4xb4_40k_pascal_context_59_480x480 import * # noqa

model.update(dict(
    decode_head=dict(
        loss_decode=[
            dict(
            type=CrossEntropyLoss,use_sigmoid=False, loss_weight=1.0),
                        dict(
            type=GTSTLoss, kd='cwd', tau=4,p_max=15,p_min=-2,pooling=False,loss_weight=1.0),
        ]
    ),
    auxiliary_head=dict(
    loss_decode=[
            dict(
            type=CrossEntropyLoss,use_sigmoid=False, loss_weight=0.4),
                        dict(
            type=GTSTLoss, kd='cwd', tau=4,p_max=15,p_min=-2,pooling=False,loss_weight=0.4),
        ]
    )))

work_dir='autodl-tmp/lhf/HRDKD-master/work_dirs_pascal_context/gtst_pspnet_r18_d8_4xb4_40k_pascal_context_59_480x480'