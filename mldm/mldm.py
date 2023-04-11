import einops
import torch
import torch as th
import torch.nn as nn

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
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.attention import SpatialTransformer
from dataset import Custom_Val_Dataset
from torch.utils.data import DataLoader
from cldm.model import load_state_dict

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
    
    def store_dataloaders(self, train_dataloader, val_dataloader):
        self.train_dataloader_log = train_dataloader
        self.val_dataloader_log = val_dataloader

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
        log["train_batch_text"] = log_txt_as_img((512, 512), train_batch_samples[self.cond_stage_key],size=16).to('cpu')

        print("---------------Logging Validation Samples---------------")
        val_batch_samples = next(iter(self.val_dataloader_log))
        val_samples, val_gt = get_samples(val_batch_samples, N)
        log["val_batch_samples"] = torch.cat((val_samples.to('cpu'), val_gt.to('cpu')), dim=0)
        log["val_batch_text"] = log_txt_as_img((512, 512), val_batch_samples[self.cond_stage_key],size=16).to('cpu')
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

        # print all keys
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
        


    