import sys
sys.path.append("/home/phebbar/Documents/ControlNet")
from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset, MySmallDataset
from dataset import Custom_Train_Dataset, Custom_Val_Dataset, Custom_FID_Dataset
from mldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import numpy as np
from cldm.ddim_hacked import DDIMSampler


version = 221

model = create_model('../models/cldm_v15.yaml').cpu()
load_path = "/lightning_logs/version_" + str(version) + "/checkpoints/epoch=0-step=0.ckpt"
model.load_state_dict(load_state_dict('../models/control_sd15_openpose.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


cond = {"c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
shape = (4, H // 8, W // 8)

if config.save_memory:
    model.low_vram_shift(is_diffusing=True)

model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, cond, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=un_cond)

if config.save_memory:
    model.low_vram_shift(is_diffusing=False)

x_samples = model.decode_first_stage(samples)
x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

results = [x_samples[i] for i in range(num_samples)]
