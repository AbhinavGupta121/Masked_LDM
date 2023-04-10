from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# pytorch seed everything
seed_everything(0)

apply_openpose = OpenposeDetector()

model = create_model('./models/mldm_v15.yaml').cpu()
"""
Model has 4 submodules:
model -> stable diffusion model
first_stage_model -> AutoencoderKL used in stable diffusion model
cond_stage_model -> CLIP Embedding
control_model -> the controlnet network
"""

# model.load_state_dict(load_state_dict('./models/control_sd15_openpose.pth', location='cuda'))
resume_path = './models/control_sd15_openpose.pth'
model.load_model_default(resume_path, location='cuda')
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image) # convert to RGB
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution) # converts min dimension to image_resolution, keeping aspect ratio
        print("Input_Image shape: ", img.shape)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST) # from openpose detector

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        # control is the output of openpose detector

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False) # Shifts the model to low VRAM mode for CLIP embedding (cond_stage_model) and first_stage_model

        #gets learned embedding from CLIP
        cond = {"c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        # Shapes
        # c_concat cond:     torch.Size([1, 3, H_resize, W_resize]) min(H_resize, W_resize) = image_resolution
        # c_crossattn cond:  torch.Size([1, 77, 768]) 77 is the number of tokens in the prompt text, 768 is the embedding dimension


        shape = (4, H // 8, W // 8) # shape of the latent space - 4,96,64
        print(shape)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True) # Shifts the model to low VRAM mode for diffusion (model and control_model)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        # control scales are defined for each parallel block of controlnet, increasing from 0.825**12 to 1.0
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        print(samples.shape) #(1, 4, 96, 64) in latent space

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False) # Shifts the model to low VRAM mode for CLIP embedding (cond_stage_model) and first_stage_model
        
        x_samples = model.decode_first_stage(samples)
        print("x_samples shape: ", x_samples.shape) # torch.Size([1, 3, 768, 512]) output between -1 and 1
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results

num_samples = 1
image_resolution = 512
strength = 1.0
guess_mode = False
detect_resolution = 512
ddim_steps = 20
scale = 9.0
seed = -1
eta = 0.0
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
a_prompt = ""
n_prompt = ""
img = cv2.imread('test_imgs/human.png')
prompt = "a man jumping on a field"
output = process(img, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
# save the output (after converting it from BGR to RGB)
cv2.imwrite('test_outputs/human_output.png', cv2.cvtColor(output[0], cv2.COLOR_BGR2RGB))
cv2.imwrite('test_outputs/human_output_1.png', cv2.cvtColor(output[1], cv2.COLOR_BGR2RGB))




# ControlLDM is top level model
# It has the following components:
# - ControlNet: 
# - ControlledUnetModel
# - AutoencoderKL
# - FrozenCLIPEmbedder