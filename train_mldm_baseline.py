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
# TODO for loss:
# - Code for 1 step prediction
# - Mask is also there in the GT now, so multiply the loss by the mask

# TODO for logging:
# - log model weights more frequently

#TODO: @abhinav - Go through train fit. At a DDPM step, get UNET output, call ddim function for calculating pred_x0. 
# Calculate VGG loss using pred_x0 and GT image masks.

class CustomCheckpointCallback(pl.callbacks.ModelCheckpoint):
    # save the model every n steps
    def __init__(self, save_every_n_steps):
        super().__init__(dirpath='checkpoints', filename='model', save_top_k=1, monitor=None)
        self.save_every_n_steps = save_every_n_steps
    
    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.save_every_n_steps == 0:
            super().on_batch_end(trainer, pl_module)

def main():
    # set global seed
    pl.seed_everything(42)
    # set numpy global seed
    np.random.seed(42)
    
    # Configs
    resume_path = './models/control_sd15_openpose.pth' #start from openpose pretrained model
    batch_size = 1
    logger_freq = 300 # log images frequency
    fid_logger_freq = 5000 # log fid frequency
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False
    calculate_fid = True
    save_model_every_n_steps = 10000
    model_loss_type = 'ddpm' # no loss change for baseline

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/mldm_v15.yaml').cpu()
    model.load_model_default(resume_path) # load controlnet with stable diffusion weights copied.
    model.control_model.del_input_hint_block() # delete the input hint block as we don't need it
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.calculate_fid = calculate_fid
    model.model_loss_type = model_loss_type

    # Misc
    logger = ImageLogger(batch_frequency=logger_freq, fid_frequency=fid_logger_freq, train_batch_size=batch_size)
    checkpointer= CustomCheckpointCallback(save_every_n_steps=save_model_every_n_steps)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpointer])
    # can pass resume_from_checkpoint=resume_path to resume training

    train_dataloader = DataLoader(Custom_Train_Dataset(), num_workers=24, batch_size=batch_size, shuffle=True)
    train_dataloader_log = DataLoader(Custom_Train_Dataset(), num_workers=24, batch_size=batch_size, shuffle=True)
    val_dataloader_log = DataLoader(Custom_Val_Dataset(), num_workers=24, batch_size=batch_size, shuffle=True)
    val_dataloader_fid = DataLoader(Custom_FID_Dataset(), num_workers=24, batch_size=batch_size, shuffle=True)
    model.store_dataloaders(train_dataloader_log, val_dataloader_log, val_dataloader_fid)

    # Train!
    trainer.fit(model, train_dataloader) 
    # Calls the training_step function in model class (in the ddpm.py file)
    # The model is of type ControlLDM which inherits from LatentDiffusion
    # When .fit() is called, the functions of the model class are called in this format:
    # training_step -> shared_step -> forward -> self.p_losses -> apply_model 
    # The ControlLDM class overloads the apply_model() function (hence the loss remains same as stable diffusion)

    # Function Documentation:
    # apply_model -> calls forward functions of diffusion_model(ControlUnetModel) and control_model(ControlNet)

    # Logging:
    # The logger callback is called  

if __name__ == '__main__':
    main()
