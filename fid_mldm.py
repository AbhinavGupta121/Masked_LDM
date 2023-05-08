import sys
sys.path.append("/home/phebbar/Documents/ControlNet")
from cldm_scripts.share import *

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
from cleanfid import fid

# global config
gpu_id = 0
fid_batch_size = 8
fid_num_samples = 1000
model_identifier = "stable_diffusion"
use_control= False
version = None
gt_path = '/home/phebbar/Documents//cocoapi/coco/person/images/train2017/'

def create_model_and_sampler(version, use_control=True):
    model = create_model('models/mldm_v15.yaml').cpu()
    model.use_control = use_control
    if(version == None): # load default controlnet weights with 0 convolutional layers if no version is specified
        model.load_model_default('./models/control_sd15_openpose.pth') # load controlnet with stable diffusion weights copied.
        model.control_model.del_input_hint_block() # delete the input hint block as we don't need it
    else:
        model.control_model.del_input_hint_block() # delete the input hint block as we don't need it
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

def model_sample(prompt, n_prompt, model, num_samples, ddim_sampler, ddim_steps, strength=1, guess_mode=True):
    cond = {"c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
    uc_cross = model.get_unconditional_conditioning(num_samples)
    uc_full = {"c_crossattn": [uc_cross]}

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    (4, 64, 64), cond, verbose=False, eta=0,
                                                    unconditional_guidance_scale=9,
                                                    unconditional_conditioning=uc_full)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    return results

def model_sample_batch(batch, model, sampler, sampler_config):
    log = dict()
    num_samples = batch['jpg'].shape[0]
    z, c = model.get_input(batch, model.first_stage_key, bs=fid_batch_size)
    c = c["c_crossattn"][0][:fid_batch_size]
    uc_cross = model.get_unconditional_conditioning(fid_batch_size)
    uc_full = {"c_crossattn": [uc_cross]}
    cond={"c_crossattn": [c]}

    # get samples using unconditional guidance
    samples, intermediates = sampler.sample(sampler_config["ddim_steps"], num_samples,
                                                    (4, 64, 64), cond, verbose=False, eta=0,
                                                    unconditional_guidance_scale=9,
                                                    unconditional_conditioning=uc_full)
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    return results

def compute_fid(gt_path, gen_path, custom_name="fid_stats_coco",mode = "clean", device="cpu"):
    """ Compute FID score for a given dataset and generator path"""
    print("DEVICE FOR FID:", device)
    if(fid.test_stats_exists(custom_name, mode=mode)):
        print("FID stats found")
        score = fid.compute_fid(gen_path, dataset_name=custom_name, mode="clean", dataset_split="custom", device = device,use_dataparallel=False)
    else:
        print("Computing FID stats for Custom GT data")
        fid.make_custom_stats(custom_name, gt_path, mode="clean", device = device)
        score = fid.compute_fid(gen_path, dataset_name=custom_name, mode="clean", dataset_split="custom", device = device,use_dataparallel=False)

    return score

def calc_fid(model, sampler, strength=1, num_samples=1000):
    print("---------------Calculating FID---------------") 
    gen_path_batch = os.path.join("FID", model_identifier)
    # make directory if it doesn't exist
    if not os.path.exists(gen_path_batch):
        os.makedirs(gen_path_batch)
    custom_dataloader = DataLoader(Custom_FID_Dataset(num_samples), num_workers=24, batch_size=fid_batch_size, shuffle=False)
    model.control_scales = [strength] * 13 
    model.eval()
    count = -1
    # create a samples_cfg_prompts.txt file in gen_path_batch
    if not os.path.exists(os.path.join(gen_path_batch, "samples_cfg_prompts.txt")):
        with open(os.path.join(gen_path_batch, "samples_cfg_prompts.txt"), "a") as f:
            f.write("Count, Prompt\n")

    with torch.no_grad():
        for batch_idx_fid, batch_fid in enumerate(custom_dataloader):
            images = model_sample_batch(batch_fid, model, sampler, {"ddim_steps":50})
            text = batch_fid["txt"]
            for i in range(len(images)):
                count = count + 1
                img = Image.fromarray(images[i])
                img.save(gen_path_batch + f"/{count}.png")
                with open(os.path.join(gen_path_batch, "samples_cfg_prompts.txt"), "a") as f:
                    f.write(str(count)+","+ text[i]+"\n")

    score = compute_fid(gt_path, gen_path_batch, device = model.device)
    print('FID:', score)

def main():
    model, ddim_sampler = create_model_and_sampler(version, use_control=use_control)
    if(model == None):
        print("Model not found")
        return
    calc_fid(model, ddim_sampler, num_samples=fid_num_samples)
    
if __name__ == "__main__":
    device = "cuda"+":"+str(gpu_id)
    torch.cuda.set_device(device)

    pl.seed_everything(42)
    np.random.seed(42)
    fid = compute_fid(gt_path, "./image_log/version294/fid_val/step74940", device = "cuda:0")
    print("FID", fid)
    # main()
