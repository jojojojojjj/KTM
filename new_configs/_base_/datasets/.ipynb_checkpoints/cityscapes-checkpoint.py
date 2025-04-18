from mmseg.datasets import CityscapesDataset
from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from mmcv.transforms.processing import Resize, RandomResize, RandomFlip, TestTimeAug
from mmseg.datasets.transforms.transforms import RandomCrop, PhotoMetricDistortion
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmengine.dataset.sampler import InfiniteSampler, DefaultSampler
from mmseg.evaluation.metrics import IoUMetric
# dataset settings
dataset_type = CityscapesDataset
data_root = '/root/autodl-tmp/data/cityscapes/'
crop_size = (512, 1024)
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomResize,
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PhotoMetricDistortion),
    dict(type=PackSegInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations),
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
            ], [dict(type=LoadAnnotations)], [dict(type=PackSegInputs)]
        ])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type=IoUMetric, iou_metrics=['mIoU'])
test_evaluator = val_evaluator
