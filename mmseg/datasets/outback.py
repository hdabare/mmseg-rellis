from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class OUTBACKDataset(BaseSegDataset):

    METAINFO = dict(
        classes=("BACKGROUND", "grass tree", "pole", "tree", "leaves", "fence net", "log", "grass", 
            "road sign", "small branch", "gravel", "ground", "horizon", "roots", "sky", 
            "delineator", "rock",),
        palette=[[0, 0, 0], [140, 255, 25], [140, 25, 255], [255, 197, 25], [25, 255, 82], 
           [25, 82, 255], [255, 25, 197], [255, 111, 25], [226, 255, 25], [54, 255, 25],
          [25, 255, 168], [25, 168, 255], [54, 25, 255], [226, 25, 255],
          [255, 25, 111], [255, 68, 25], [255, 154, 25]]
)

    def __init__(self, **kwargs):
        super(OUTBACKDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_orig.png',
            **kwargs)