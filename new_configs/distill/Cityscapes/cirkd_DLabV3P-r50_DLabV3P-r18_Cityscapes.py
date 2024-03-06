from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder,ModuleInputsRecorder
from mmrazor.models.architectures.connectors import ConvModuleConnector
from razor.models.losses import CriterionKD, StudentSegContrast, CriterionMiniBatchCrossImagePair
with read_base():
    from ..._base_.default_runtime import * # noqa
    from ..._base_.datasets.cityscapes import * # noqa
    from ..._base_.schedules.schedule_40k import * # noqa

teacher_cfg_path = "autodl-tmp/lhf/HRDKD-master/new_configs/deeplabv3+/deeplabv3plus_r50_d8_4xb2_40k_cityscapes_512x1024.py"  # noqa: E501
student_cfg_path = 'autodl-tmp/lhf/HRDKD-master/new_configs/deeplabv3+/deeplabv3plus_r18b_d8_4xb2_40k_cityscapes_512x1024.py'  # noqa: E501
teacher_ckpt = "autodl-tmp/lhf/HRDKD-master/ckpt/dpv3p_r50_cityscapes/best_mIoU_78-61_iter_40000.pth"

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
            gt=dict(type=ModuleInputsRecorder, source='decode_head.loss_decode'),
        ),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg'),
            feats=dict(type=ModuleInputsRecorder, source='decode_head.conv_seg'),
        ),
        distill_losses=dict(
            loss_kd=dict(
                type=CriterionKD,loss_weight=1),
            loss_memory=dict(
                type=StudentSegContrast,
                num_classes=19,
                s_channels=128,
                t_channels=512,
                loss_weight=1.0),
            loss_minibatch=dict(
                type=CriterionMiniBatchCrossImagePair,temperature=1,pooling=True,loss_weight=1),
        ),
        loss_forward_mappings=dict(
            loss_kd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
            ),
            loss_memory=dict(
                s_feats=dict(from_student=True, recorder='feats',data_idx=0), 
                t_feats=dict(from_student=False, recorder='feats',data_idx=0),
                labels=dict(from_student=True, recorder='gt',data_idx=1), 
                predict=dict(from_student=True, recorder='logits'),
            ),
            loss_minibatch=dict(
            feat_S=dict(from_student=True, recorder='feats',data_idx=0),
            feat_T=dict(from_student=False, recorder='feats',data_idx=0),
            )
        ),
        ))
work_dir='autodl-tmp/lhf/HRDKD-master/work_dir/cirkd_DLabV3P-r50_DLabV3P-r18_Cityscapes'