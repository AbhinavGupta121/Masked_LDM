from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from mldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = './models/control_sd15_openpose.pth' #start from openpose pretrained model
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/mldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.control_model.del_input_hint_block() # delete the input hint block as we don't need it
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

# Train!
trainer.fit(model, dataloader) 
# Calls the training_step function in model class (in the ddpm.py file)
# The model is of type ControlLDM which inherits from LatentDiffusion
# When .fit() is called, the functions of the model class are called in this format:
# training_step -> shared_step -> forward -> self.p_losses -> apply_model 
# The ControlLDM class overloads the apply_model() function (hence the loss remains same as stable diffusion)

# Function Documentation:
# apply_model -> calls forward functions of diffusion_model(ControlUnetModel) and control_model(ControlNet)

# Logging:
# The logger callback is called  
