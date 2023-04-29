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
import glob
import torch
import einops
import os
from PIL import Image

gpu_id = 1
device = "cuda"+":"+str(gpu_id)
torch.cuda.set_device(device)

def create_model_and_sampler(version, use_control=True):
    model = create_model('models/mldm_v15.yaml').cpu()
    model.control_model.del_input_hint_block() # delete the input hint block as we don't need it
    model.use_control = use_control
    file_list = glob.glob(f'lightning_logs/version_{version}/checkpoints/*.ckpt')
    if(len(file_list) == 0):
        print("No checkpoint file found")
        return None, None
    checkpoint_path = file_list[0]
    # load model from checkpoint using pytorch lightning
    state_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler

def model_sample(prompt, n_prompt, model, num_samples, ddim_sampler, ddim_steps, eta, unconditional_guidance_scale, strength=1, guess_mode=True):
    cond = {"c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
    un_cond = {"c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    shape = (4, 64, 64)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    return results

def main():
    prompt = "a person jumping on a farm"
    n_prompt = "more than two hands, more than two legs, distorted face"
    use_control= True
    guess_mode = False
    
    unconditional_guidance_scale = 9.0
    num_samples = 1
    ddim_steps = 50
    eta = 0
    control_strength = 1

    model, ddim_sampler = create_model_and_sampler(225, use_control=use_control)
    if(model == None):
        print("Model not found")
        return

    results = model_sample(prompt, n_prompt, model, num_samples, ddim_sampler, ddim_steps, eta, unconditional_guidance_scale, control_strength, guess_mode)
    print(results[0].shape) # 512, 512, 3
    # save results
    for i in range(len(results)):
        img = results[i]
        img = Image.fromarray(img)
        img.save(f"test_outputs/{i}.png")
    
if __name__ == "__main__":
    main()
