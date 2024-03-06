from mmseg.datasets import PascalContextDataset59
from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from mmcv.transforms.processing import Resize, RandomResize, RandomFlip, TestTimeAug,Pad
from mmseg.datasets.transforms.transforms import RandomCrop, PhotoMetricDistortion
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmengine.dataset.dataset_wrapper import ConcatDataset
from mmengine.dataset.sampler import InfiniteSampler, DefaultSampler
from mmseg.evaluation.metrics import IoUMetric

# dataset settings
dataset_type = PascalContextDataset59
data_root = 'autodl-tmp/data/OpenDataLab___VOC2010/raw/Images/VOCdevkit/VOC2010/'

img_scale = (520, 520)
crop_size = (480, 480)

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations, reduce_zero_label=True),
    dict(
        type=RandomResize,
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PhotoMetricDistortion),
    dict(type=PackSegInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations, reduce_zero_label=True),
    dict(type=PackSegInputs)
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),
    dict(
        type=TestTimeAug,
        transforms=[
            [
                dict(type=Resize, scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type=RandomFlip, prob=0., direction='horizontal'),
                dict(type=RandomFlip, prob=1., direction='horizontal')
            ], [dict(type=LoadAnnotations)], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type=IoUMetric, iou_metrics=['mIoU'])
test_evaluator = val_evaluator