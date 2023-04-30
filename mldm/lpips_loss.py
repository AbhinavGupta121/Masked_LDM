import einops
import os
import torch
import torch as th
import torch.nn as nn
import types
import lpips
import numpy as np
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from cldm.cldm import ControlLDM, ControlNet
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.attention import SpatialTransformer
from dataset import Custom_Val_Dataset
from torch.utils.data import DataLoader
from cldm.model import load_state_dict
import matplotlib.pyplot as plt


def check_loss_input(im0, im1, w):
    """ im0 is out and im1 is target and w is mask"""
    assert list(im0.size())[2:] == list(im1.size())[2:], 'spatial dim mismatch'
    if w is not None:
        assert list(im0.size())[2:] == list(w.size())[2:], 'spatial dim mismatch'

    if im1.size(0) != 1:
        assert im0.size(0) == im1.size(0)

    if w is not None and w.size(0) != 1:
        assert im0.size(0) == w.size(0)
    return

class Masked_LPIPS_Loss(nn.Module):
    def __init__(self, net='vgg', device='cuda', precision='float'):
        """ LPIPS loss with spatial weighting """
        super(Masked_LPIPS_Loss, self).__init__()
        self.lpips = lpips.LPIPS(net=net, spatial=True).eval()
        self.lpips = self.lpips.to(device)
        if precision == 'half':
            self.lpips.half()
        elif precision == 'float':
            self.lpips.float()
        elif precision == 'double':
            self.lpips.double()
        return

    def forward(self, im0, im1, w=None):
        """ ims have dimension BCHW while mask is B1HW """
        check_loss_input(im0, im1, w)
        # lpips takes the sum of each spatial map
        loss = self.lpips(im0, im1)
        if w is not None:
            n = torch.sum(loss * w, [1, 2, 3])
            d = torch.sum(w, [1, 2, 3])
            loss = n / d
        return loss

    def __call__(self, im0, im1, w=None):
        return self.forward(im0, im1, w)


