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

def isfunction(object):
    """Return true if the object is a user-defined function.

    Function objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this function was defined
        __code__        code object containing compiled function bytecode
        __defaults__    tuple of any default values for arguments
        __globals__     global namespace in which this function was defined
        __annotations__ dict of parameter annotations
        __kwdefaults__  dict of keyword only parameters with defaults"""
    return isinstance(object, types.FunctionType)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class MaskControlNet(ControlNet):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        # remove the self.input_hint_block
        super().__init__(image_size, in_channels, model_channels, hint_channels, num_res_blocks, attention_resolutions, dropout, channel_mult, conv_resample, dims, use_checkpoint, use_fp16, num_heads, num_head_channels, num_heads_upsample, use_scale_shift_norm, resblock_updown, use_new_attention_order, use_spatial_transformer, transformer_depth, context_dim, n_embed, legacy, disable_self_attentions, num_attention_blocks, disable_middle_self_attn, use_linear_in_transformer)

    def del_input_hint_block(self):
        del self.input_hint_block

    def forward(self, x, timesteps, context, **kwargs):
        # context is the conditioning from text
        # overload the forward function to break the connection to 
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        return outs

class MaskControlLDM(ControlLDM):
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        # call the grandparent class constructor
        # removed the control_key argument (as no control key is needed for this model)
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        del self.control_key
    
    def store_dataloaders(self, train_dataloader, val_dataloader, val_dataloader_fid):
        self.train_dataloader_log = train_dataloader
        self.val_dataloader_log = val_dataloader
        self.val_dataloader_fid = val_dataloader_fid

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = LatentDiffusion.get_input(self, batch=batch, k=self.first_stage_key, *args, **kwargs)
        # c is the conditioning from text, x is latent
        self.ddim_shape = (self.channels, x.shape[2], x.shape[3])
        return x, dict(c_crossattn=[c])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        This function overloads the apply_model function of the ControlLDM class.
        Changes: removed the hint argument from controlnet call
        """
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1) # output is tensor of shape (1, 77, 768)
        control = self.control_model(x=x_noisy, timesteps=t, context=cond_txt) #removed the hint argument
        control = [c * scale for c, scale in zip(control, self.control_scales)] # list of len 13, each element is control to be applied at different unet layers
        # for i in range(len(control)):
            # control[i] = torch.zeros_like(control[i])
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control) # ControlUnet
        return eps

    @torch.no_grad() 
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        """
        Called by the logger class, which in turn is called by pytorch lightning trainer.
        """
        use_ddim = ddim_steps is not None

        log = dict()
        text = dict()
        # log["reconstruction"] = self.decode_first_stage(z) # test for autoencoder
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        
        def get_samples(new_batch, N):
            z, c = self.get_input(new_batch, self.first_stage_key, bs=N)
            c = c["c_crossattn"][0][:N]
            N = min(z.shape[0], N)
            uc_cross = self.get_unconditional_conditioning(N) # null text conditioning
            uc_full = {"c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_crossattn": [c]},
                                                batch_size=N, ddim=use_ddim,
                                                ddim_steps=ddim_steps, eta=ddim_eta,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc_full,
                                                )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            # print(new_batch['jpg'].permute(0, 3, 1, 2))
            return x_samples_cfg, new_batch['jpg'].permute(0, 3, 1, 2)
        
        print("---------------Logging Train Samples---------------")
        train_batch_samples = next(iter(self.train_dataloader_log))
        train_samples, train_gt= get_samples(train_batch_samples, N)
        log["train_batch_samples"] = torch.cat((train_samples.to('cpu'),train_gt.to('cpu')), dim=0)
        # log["train_batch_text"] = log_txt_as_img((512, 512), train_batch_samples[self.cond_stage_key],size=16).to('cpu')
        text["train_batch_text"] = train_batch_samples[self.cond_stage_key]    

        print("---------------Logging Validation Samples---------------")
        val_batch_samples = next(iter(self.val_dataloader_log))
        val_samples, val_gt = get_samples(val_batch_samples, N)
        log["val_batch_samples"] = torch.cat((val_samples.to('cpu'), val_gt.to('cpu')), dim=0)
        # log["val_batch_text"] = log_txt_as_img((512, 512), val_batch_samples[self.cond_stage_key],size=16).to('cpu')
        text["val_batch_text"] = val_batch_samples[self.cond_stage_key]
        return log, text

    def log_loss(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        """
        Called by the logger class, which in turn is called by pytorch lightning trainer.
        """
        use_ddim = ddim_steps is not None
        log = dict()
        def get_loss(new_batch, N):
            z, c = self.get_input(new_batch, self.first_stage_key, bs=N)
            c = c["c_crossattn"][0][:N]
            N = min(z.shape[0], N)
            uc_cross = self.get_unconditional_conditioning(N) # null text conditioning
            uc_full = {"c_crossattn": [uc_cross]}
            t = torch.randint(0, self.ddpm_mask_thresh, (z.shape[0],), device=self.device).long()
            new_batch['mask'] = new_batch['mask'].to(self.device)
            new_batch['jpg'] = new_batch['jpg'].to(self.device)
            _, loss_dict, x_0_pred= self.mask_aware_loss(z, cond={"c_crossattn": [c]}, t=t, mask = new_batch['mask'], x0_gt=new_batch['jpg'])
            x_noisy = self.decode_first_stage(z)
            # print(new_batch['jpg'].permute(0, 3, 1, 2))
            return x_noisy, loss_dict, t, x_0_pred, new_batch['jpg'].permute(0, 3, 1, 2), new_batch['mask'].permute(0, 3, 1, 2) 
        
        print("---------------Logging Train Loss Samples---------------")
        prefix = 'train' if self.training else 'val'
        train_batch_sample = next(iter(self.train_dataloader_log))
        x_noisy, loss_dict, timestep, x_0_pred, train_gt, mask_gt = get_loss(train_batch_sample, N)
        log['x_noisy'] = x_noisy.to('cpu')
        log["x_text"] = train_batch_sample[self.cond_stage_key]    
        log['loss_sd'] = loss_dict[f'{prefix}/loss_sd'].to('cpu')
        log['loss_mask'] = loss_dict[f'{prefix}/loss_mask'].to('cpu')
        log['timestep'] = timestep.to('cpu')
        log['x_0_pred'] = x_0_pred.to('cpu')
        log['train_gt'] = train_gt.to('cpu')
        log['mask_gt'] = mask_gt.to('cpu')
        # print all items in log
        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = self.ddim_shape
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @torch.no_grad() 
    def log_images_fid(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        """
        Called by the logger class, which in turn is called by pytorch lightning trainer.
        """
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c = c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        # get samples using unconditional guidance
        uc_cross = self.get_unconditional_conditioning(N)
        uc_full = {"c_crossattn": [uc_cross]}
        samples_cfg, _ = self.sample_log(cond={"c_crossattn": [c]},
                                            batch_size=N, ddim=use_ddim,
                                            ddim_steps=ddim_steps, eta=ddim_eta,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc_full,
                                            )
        x_samples_cfg = self.decode_first_stage(samples_cfg)
        log[f"samples_cfg"] = x_samples_cfg
        return log

    def load_model_default(self, resume_path, location='cpu'):
        """
        Loads the model with default stable diffusion parameters loaded in self.control_model
        """
        dict = load_state_dict(resume_path, location=location)
        # load all keys that start with model.diffusion_model
        dict_diffusion = {}; dict_cond_stage = {}; dict_first_stage = {}; dict_control_model = {}

        count1 = 0; count2 = 0; count3 = 0; count4 = 0
        for key, value in dict.items():
            # if key begins with model. 
            if(key.startswith('model.diffusion_model')):
                #load the value in model.diffusion_model
                key = key.replace("model.diffusion_model.", "")
                dict_diffusion[key] = value
                count1 = count1 + 1

                if(key.startswith("time_embed") or key.startswith("input_blocks") or key.startswith("middle_block")): # load into control_model
                    dict_control_model[key] = value
                    count2 = count2 + 1
                    # NOTE this does not load the inpu_hint_block and zero_conv blocks

            elif(key.startswith('cond_stage_model')):
                key = key.replace("cond_stage_model.", "")
                dict_cond_stage[key] = value
                count3 = count3 + 1

            elif(key.startswith('first_stage_model')):
                key = key.replace("first_stage_model.", "")
                dict_first_stage[key] = value
                count4 = count4 + 1
    
        self.model.diffusion_model.load_state_dict(dict_diffusion, strict=True)
        self.control_model.load_state_dict(dict_control_model, strict=False)
        self.cond_stage_model.load_state_dict(dict_cond_stage, strict=True)
        self.first_stage_model.load_state_dict(dict_first_stage, strict=True)
        
    def forward(self, x, c, x0_gt = None, mask = None, *args, **kwargs):
        """Forward pass of the model. Both MSE or mask-aware loss can be used."""
        t = torch.randint(0,self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        if(self.model_loss_type == 'ddpm'):
            return self.p_losses(x, c, t, *args, **kwargs)
        elif(self.model_loss_type =='mask'): #mask-aware loss
            assert mask is not None
            assert x0_gt is not None
            # self.inference_samples(x, c, t,*args, **kwargs)
            loss, loss_dict, _ = self.mask_aware_loss(x, c, t, mask, x0_gt= x0_gt)
            return loss, loss_dict

    def get_x0(self, xt, t, cond, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * noise
        )

    def mask_aware_loss(self, x_start, cond, t, mask, x0_gt, noise=None):
        """
        Calculate the Mask Aware Loss
        """
        #x_start is the latent vector of GT image. It'll be noised in subsequent foraward diffusion process by qsample
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # Calculate the Normal Stable Diffusion loss
        model_output = self.apply_model(x_noisy, t, cond) # apply_model has been overloaded in cldm.

        #Target depends on the parameterization uses. Either predict eps directly or x0, or v.
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss_sd': loss})

        # Mask Aware Loss
        if(t > self.ddpm_mask_thresh):
            loss_dict.update({f'{prefix}/loss_mask': 0})
            x0_pred_img = torch.zeros_like(x0_gt)
        else:
            loss_fn = lpips.LPIPS(net='vgg',verbose=False).to(x0_gt.device)
            x0_pred = self.get_x0(x_noisy, t, cond, model_output)
            x0_pred_img = self.decode_first_stage(x0_pred)
            x0_gt = x0_gt.permute(0, 3, 1, 2)
            mask = mask.permute(0, 3, 1, 2)
            # print devices
            # print(x0_pred_img.device, x0_gt.device, mask.device)
            # print(x0_pred_img.shape, x0_gt.shape, mask.shape)
            loss_mask = loss_fn.forward(x0_pred_img*mask, x0_gt*mask).mean()
            loss_dict.update({f'{prefix}/loss_mask': loss_mask})
            # loss = loss_mask
            loss = (1-self.mask_weight)*loss + self.mask_weight * loss_mask
        
        loss_dict.update({f'{prefix}/loss': loss})
        return loss, loss_dict, x0_pred_img

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, x0_gt = batch["jpg"], mask = batch["mask"])
        return loss

    def training_step(self, batch, batch_idx):
        """
        Samples a random timestep and trains on that timestep
        """
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss



    # @torch.no_grad()
    # def inference_samples(self, x_start,cond, t, ddim_steps=50,noise=None, **kwargs):

    #     noise = default(noise, lambda: torch.randn_like(x_start))
    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

    #     loss_dict = {}
    #     prefix = 'train' if self.training else 'val'

    #     log={}
    #     # ddim_sampler = DDIMSamplerOneStep(self)
    #     shape = (self.channels, self.image_size, self.image_size)
    #     batch_size = x_start.shape[0]
    #     x_samples = self.get_x0(x_noisy, t, cond)
    #     x_samples = self.decode_first_stage(x_samples.to(self.device)).detach().cpu().numpy()
        
    #     for i in range(batch_size):
    #         x_sample = x_samples[i]
    #         x_sample = ((np.clip(x_sample, -1., 1.) + 1)/2)*255
    #         x_sample = x_sample.astype(np.uint8) 
    #         # x_sample = np.squeeze(x_sample, axis=0)
    #         x_sample = x_sample.transpose((1,2,0))
    #         filename1 = f"x0_{i}.png"
    #         folder =  '/home/phebbar/Documents/ControlNet_VLR/intermediate_res'
    #         print("x_sample shape: ", x_sample.shape)
    #         plt.imsave(os.path.join(folder, filename1),x_sample)

    #     filename2 = f"x_start.png"
    #     filename3 = f"x_noisy.png"
        
    #     #save intermediates images in directory
    #     # for key in intermediates.keys():
    #     #     for i in range(len(intermediates[key])):
    #     #         filename1 = f"intermediate_{key}_{i}.png"
    #     #         folder =  '/home/phebbar/Documents/ControlNet_VLR/intermediate_res'
    #     #         img = intermediates[key][i]
                
    #     #         x_sample = self.decode_first_stage(img.to(self.device))
    #     #         x_sample = x_sample.detach().cpu().numpy()

    #     #         #save the image in folder with filename
    #     #         x_sample = ((np.clip(x_sample, -1., 1.) + 1)/2)*255
    #     #         x_sample = x_sample.astype(np.uint8) 
    #     #         #squeeze first dim in x_sample
    #     #         x_sample = np.squeeze(x_sample, axis=0)
    #     #         x_sample = x_sample.transpose((1,2,0))
    #     #         plt.imsave(os.path.join(folder, filename1),x_sample)

    #     out1 =self.decode_first_stage(x_start)[0].detach().cpu().numpy()
    #     out1 = ((np.clip(out1, -1., 1.) + 1)/2)*255
    #     out1 = out1.astype(np.uint8) 
    #     out1 = out1.transpose((1,2,0))

    #     out2= self.decode_first_stage(x_noisy)[0].detach().cpu().numpy()
    #     out2 = ((np.clip(out2, -1., 1.) + 1)/2)*255
    #     out2 = out2.astype(np.uint8) 
    #     out2 = out2.transpose((1,2,0))

    #     plt.imsave(os.path.join(folder, filename2),out1)
    #     plt.imsave(os.path.join(folder, filename3),out2)

    