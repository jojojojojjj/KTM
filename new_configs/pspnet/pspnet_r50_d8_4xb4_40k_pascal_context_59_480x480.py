from mmengine.config import read_base
from torch.optim import SGD
from mmengine.optim import OptimWrapper
with read_base():
    from .._base_.models.pspnet_r50_d8 import * # noqa 
    from .._base_.datasets.pascal_context_59 import * # noqa
    from .._base_.default_runtime import * # noqa
    from .._base_.schedules.schedule_40k import * # noqa

crop_size = (480, 480)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=59),
    auxiliary_head=dict(num_classes=59),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320))))
optimizer.update(dict(type=SGD, lr=0.004, momentum=0.9, weight_decay=0.0001))
optim_wrapper.update(dict(type=OptimWrapper, optimizer=optimizer))

work_dir='autodl-tmp/lhf/HRDKD-master/work_dirs_pascal_context/pspnet_r50_d8_4xb4_40k_pascal_context_59_480x480'