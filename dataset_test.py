from dataset import Train_Dataset, Val_Dataset, Custom_Train_Dataset, Custom_Val_Dataset
import numpy as np
import matplotlib.pyplot as plt

train_dataset = Custom_Train_Dataset()
val_dataset = Custom_Val_Dataset()
print(len(train_dataset), len(val_dataset))

item = train_dataset[129]
jpg = item['jpg']
mask = item['mask']
txt = item['txt']
print("train")
print(txt)
print(mask.shape)
print(jpg.shape)

item = val_dataset[129]
jpg = item['jpg']
mask = item['mask']
txt = item['txt']
print("val")
print(txt)
print(mask.shape)
print(mask.tolist())
print(jpg.shape)
plt.imshow(mask)
# use cv imread to read boolean mask