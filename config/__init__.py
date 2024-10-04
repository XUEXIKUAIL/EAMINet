from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, adjust_lr
# from .ranger.ranger import Ranger
# from .ranger.ranger913A import RangerVA
# from .ranger.rangerqh import RangerQH
import torch.nn as nn

def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'sunrgbd224', 'WE3DS', 'IRSeg']
    if cfg['dataset'] == 'nyuv2':
        from ..datasets.nyuv2 import NYUv2
        return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd':
        from ..datasets.sunrgbd import SUNRGBD
        from ..datasets.sunrgbd_eval import SUNRGBD2
        return SUNRGBD(cfg, mode='train'), SUNRGBD2(cfg, mode='test')

def get_model_t(cfg):

    if cfg['model_name'] == '4_Model_segb2R_mobilev2D':
        from ..EAMINet import EnDecoderModel
        return EnDecoderModel(backbone='segb2R_mobilev2D', n_classes=cfg['n_classes'])






