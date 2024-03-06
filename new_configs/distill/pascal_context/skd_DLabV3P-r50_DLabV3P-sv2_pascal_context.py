from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder,ModuleInputsRecorder
from mmrazor.models.architectures.connectors import ConvModuleConnector
from razor.models.losses import CriterionPairWiseforWholeFeatAfterPool,CriterionKD
with read_base():
    from ..._base_.default_runtime import * # noqa
    from ..._base_.datasets.pascal_context_59 import * # noqa
    from ..._base_.schedules.schedule_40k import * # noqa

teacher_cfg_path = "autodl-tmp/lhf/HRDKD-master/new_configs/deeplabv3+/deeplabv3plus_r50_d8_4xb4_40k_pascal_context_59_480x480.py"  # noqa: E501
student_cfg_path = 'autodl-tmp/lhf/HRDKD-master/new_configs/deeplabv3+/shufflenet-v2-d8_deeplabv3plus_4xb2-40k_pascal_context-480x480.py'  # noqa: E501
teacher_ckpt = "autodl-tmp/lhf/HRDKD-master/ckpt/dpv3p_r50_pascal_context/best_mIoU_49-42_iter_40000.pth"


model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt = teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg'),
            feats=dict(type=ModuleInputsRecorder, source='decode_head.conv_seg'),
        ),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg'),
            feats=dict(type=ModuleInputsRecorder, source='decode_head.conv_seg'),
        ),
        distill_losses=dict(
            loss_pi=dict(
                type=CriterionKD,
                loss_weight=1.0),
            loss_pa=dict(
                type=CriterionPairWiseforWholeFeatAfterPool,
                loss_weight=1.0),
        ),
        loss_forward_mappings=dict(
            loss_pi=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
            ),
            loss_pa=dict(
            feat_S=dict(from_student=True, recorder='feats',data_idx=0),
            feat_T=dict(from_student=False, recorder='feats',data_idx=0),
            )
        ),
        ))
work_dir='autodl-tmp/lhf/HRDKD-master/work_dir/skd_DLabV3P-r50_DLabV3P-sv2_pascal_context'