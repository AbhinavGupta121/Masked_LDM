import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from cleanfid import fid

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, fid_frequency=1000, train_batch_size=2):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.fid_frequency = fid_frequency
        self.gt_path = '/home/phebbar/Documents//cocoapi/coco/person/images/train2017/'
        self.gen_path = '/home/phebbar/Documents/ControlNet/image_log/'
        self.train_batch_size = train_batch_size

    @rank_zero_only 
    def log_local(self, save_dir, split, images, text, global_step, current_epoch, batch_idx, version):
        """
        Decorator used to run this method only on the main process (with rank 0)
        Makes a grid of images and saves it to disk
        """
        # get version of pytorch lightning logger
        root = os.path.join(save_dir, "image_log", "version" + str(version) , split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=self.train_batch_size, pad_value=1)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # [-1,1] -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
        
        # store text prompts in a csv file (text will have train_batch_text and val_batch_text)
        for k in text:
            if not os.path.exists(os.path.join(root, k+".csv")):
                with open(os.path.join(root, k+".csv"), "w") as f:
                    f.write("Global Step, Current Epoch, Batch Index, Prompt\n")
            # append text to prompts.txt in a new line with 
            with open(os.path.join(root, k+".csv"), "a") as f:
                f.write(str(global_step)+","+str(current_epoch)+","+ str(batch_idx)+","+ str(text[k])+"\n")


    def log_img(self, pl_module, batch, batch_idx, split="train"):
        """
        Called by on_train_batch_end
        Logs images to image_log/train or image_log/{split}
        Calls the log_images function of the model 
        """
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval() # set to eval mode

            with torch.no_grad():
                images, text = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images, text,
                           pl_module.global_step, pl_module.current_epoch, batch_idx, pl_module.logger.version)

            if is_train:
                pl_module.train() # restore training mode

    def calc_fid(self, pl_module, batch_idx, split="fid_val"):
        """
        Called by on_train_batch_end
        Logs images to image_log/train or image_log/{split}
        Calls the log_images function of the model 
        """
        if(pl_module.calculate_fid==False):
            return
        if (batch_idx % self.fid_frequency == 0): # log every fid_frequency batches

            print("---------------Calculating FID---------------") 
            gen_path_batch = os.path.join(self.gen_path, "version" + str(pl_module.logger.version), "fid_val", "step"+str(pl_module.global_step))
            custom_dataloader  = pl_module.val_dataloader_fid

            is_train = pl_module.training
            if is_train:
                pl_module.eval() # set to eval mode

            with torch.no_grad():
                for batch_idx_fid, batch_fid in enumerate(custom_dataloader):
                    images = pl_module.log_images_fid(batch_fid, split=split, **self.log_images_kwargs)
                    text = batch_fid["txt"]
                    for k in images:
                        N = min(images[k].shape[0], self.max_images)
                        images[k] = images[k][:N]
                        if isinstance(images[k], torch.Tensor):
                            images[k] = images[k].detach().cpu()
                            if self.clamp:
                                images[k] = torch.clamp(images[k], -1., 1.)

                        if(k=='samples_cfg'):
                            for i in range(batch_fid["jpg"].shape[0]):
                                grid = images[k][i]
                                if self.rescale:
                                    grid = (grid + 1.0) / 2.0  # [-1,1] -> 0,1; c,h,w
                                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                                grid = grid.numpy()
                                grid = (grid * 255).astype(np.uint8)
                                filename = "{}_gs-{:06}_e-{:06}_b-{:06}_idx{:03}.png".format(k, pl_module.global_step, pl_module.current_epoch, batch_idx_fid,i)
                                path = os.path.join(gen_path_batch, filename)
                                os.makedirs(os.path.split(path)[0], exist_ok=True)
                                Image.fromarray(grid).save(path)
                            
                            # create a samples_cfg_prompts.txt file in gen_path_batch
                            if not os.path.exists(os.path.join(gen_path_batch, "samples_cfg_prompts.txt")):
                                with open(os.path.join(gen_path_batch, "samples_cfg_prompts.txt"), "a") as f:
                                    f.write("Global Step, Current Epoch, Batch Index, Prompt\n")
                            for i in range(batch_fid["jpg"].shape[0]): # for each image in the batch
                                with open(os.path.join(gen_path_batch, "samples_cfg_prompts.txt"), "a") as f:
                                    f.write(str(pl_module.global_step)+","+str(pl_module.current_epoch)+","+ str(batch_idx_fid)+","+ text[i]+"\n")

            score = self.compute_fid(self.gt_path, gen_path_batch)
            print('FID:', score)
            #log using pytorch lightning
            pl_module.log('FID', score, on_step=True, on_epoch=False, prog_bar = False, logger=True, batch_size = self.train_batch_size)
            if is_train:
                pl_module.train() # restore training mode

    def compute_fid(self, gt_path, gen_path,custom_name="fid_stats_coco",mode = "clean"):

        """ Compute FID score for a given dataset and generator path"""

        if(fid.test_stats_exists(custom_name, mode=mode)):
            print("FID stats found")
            score = fid.compute_fid(gen_path, dataset_name=custom_name, mode="clean", dataset_split="custom")
        else:
            print("Computing FID stats for Custom GT data")
            fid.make_custom_stats(custom_name, gt_path, mode="clean")
            score = fid.compute_fid(gen_path, dataset_name=custom_name, mode="clean", dataset_split="custom")

        return score

    def check_frequency(self, check_idx):
        """
        Local function, called by log_img
        """
        return check_idx % self.batch_freq == 0 and check_idx > 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        Called by PyTorch Lightning when the train batch ends.        
        """
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
            self.calc_fid(pl_module, batch_idx, split="fid_val")

                    
