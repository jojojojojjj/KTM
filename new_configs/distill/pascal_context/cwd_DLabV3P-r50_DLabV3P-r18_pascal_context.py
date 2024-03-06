from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder,ModuleInputsRecorder
from mmrazor.models.architectures.connectors import ConvModuleConnector
from razor.models.losses import ChannelWiseDivergence
with read_base():
    from ..._base_.default_runtime import * # noqa
    from ..._base_.datasets.pascal_context_59 import * # noqa
    from ..._base_.schedules.schedule_40k import * # noqa

teacher_cfg_path = "autodl-tmp/lhf/HRDKD-master/new_configs/deeplabv3+/deeplabv3plus_r50_d8_4xb4_40k_pascal_context_59_480x480.py"  # noqa: E501
student_cfg_path = 'autodl-tmp/lhf/HRDKD-master/new_configs/deeplabv3+/deeplabv3plus_r18_d8_4xb4_40k_pascal_context_59_480x480.py'  # noqa: E501
teacher_ckpt = "autodl-tmp/lhf/HRDKD-master/ckpt/dpv3p_r50_pascal_context/best_mIoU_49-42_iter_40000.pth"



model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt = teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        distill_losses=dict(
            loss_cwd=dict(
                type=ChannelWiseDivergence,
                tau=4,
                loss_weight=1,),
        ),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
        ),
        ),
        ))
work_dir='autodl-tmp/lhf/HRDKD-master/work_dir/cwd_DLabV3P-r50_DLabV3P-r18_pascal_context'