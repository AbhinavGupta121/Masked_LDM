from dataset import Custom_Train_Dataset, Custom_Val_Dataset
from torch.utils.data import DataLoader

batch_size = 2
train_dataloader = DataLoader(Custom_Train_Dataset(), num_workers=1, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(Custom_Val_Dataset(), num_workers=1, batch_size=batch_size, shuffle=True)
dataset = Custom_Train_Dataset()
# iterate over the dataloader
# for i in range(0, 56000):
#     item = dataset.__getitem__(i)
#     print("batch: ", item)
#     break

for i in range(len(val_dataloader)):
    print(i)