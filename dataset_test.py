from dataset import Train_Dataset, Val_Dataset, Custom_Train_Dataset, Custom_Val_Dataset

train_dataset = Custom_Train_Dataset()
val_dataset = Custom_Val_Dataset()
print(len(train_dataset), len(val_dataset))

item = train_dataset[129]
jpg = item['jpg']
txt = item['txt']
print("train")
print(txt)
print(jpg.shape)

item = val_dataset[129]
jpg = item['jpg']
txt = item['txt']
print("val")
print(txt)
print(jpg.shape)