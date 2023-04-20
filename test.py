from dataset import Custom_Train_Dataset, Custom_Val_Dataset
from torch.utils.data import DataLoader
import os

save_dir = "/home/phebbar/Documents/ControlNet/testing"
version = 1
split = "train"
root = os.path.join(save_dir, "image_log", "version" + str(version) , split)
text = dict()
text["train_batch_text"] = "testingtrain"
text["val_batch_text"] = "testingval"

global_step = 4
current_epoch = 5
batch_idx = 6
for k in text:
    # make a .txt file called prompts.txt at root if it doesn't exist
    if not os.path.exists(os.path.join(root, k+".csv")):
        os.makedirs(root, exist_ok=True)
        print("not found")
        with open(os.path.join(root, k+".csv"), "w") as f:
            f.write("Global Step, Current Epoch, Batch Index, Prompt\n")

    # append text to prompts.txt in a new line with 
    with open(os.path.join(root, k+".csv"), "a") as f:
        f.write(str(global_step)+","+str(current_epoch)+","+ str(batch_idx)+","+ text[k]+"\n")