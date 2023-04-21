import json
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from torch.utils.data import Dataset
import random


def custom_resize(img, mask):
    # img shape is (height, width, channels)
    height, width = img.shape[:2]
    if(height < width):
        # transpose image along 0 and 1 axis
        img, mask = custom_resize(img.transpose(1, 0, 2), mask.transpose(1, 0))
        # print("transposed", img.shape, mask.shape)
        # transpose back
        img = img.transpose(1, 0, 2)
        mask = mask.transpose(1, 0)
        return img, mask
    else:
        # get the scale factor for height
        if(height > 256): # resize img if max_dim > 256
            # Determine the scaling factor to resize the image
            scale_factor = 256 / height
        else:
            # scale such that height is a multiple of 64
            # get closest multiple of 64
            dist = np.abs(np.array([64, 128, 192, 256]) - height)
            closest_multiple = (1+np.argmin(dist))*64
            scale_factor = closest_multiple / height
            # scale_factor = max( 64 / height, 128/height, 256/height)
        # Resize the image
        wi_new = int(width * scale_factor)
        hi_new = int(height * scale_factor)
        # print("raw scale", hi_new, wi_new, end=' - ')
        if(wi_new<64):
            # resize width to 64
            scale_factor_x = 64 / width
            resized_img = cv2.resize(img, None, fx=scale_factor_x, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(mask, None, fx=scale_factor_x, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            # print("small", resized_img.shape, resized_mask.shape)

        elif(wi_new%64 < 10 or wi_new%64 > 54): #the width is very near to a multiple of 64, so just resize
            # print("close", end=' - ')
            # get nearest multiple of 64
            dist = np.abs(np.array([64, 128, 192, 256]) - wi_new)
            closest_multiple = (1+np.argmin(dist))*64
            scale_factor_x = closest_multiple / width
            resized_img = cv2.resize(img, None, fx=scale_factor_x, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(mask, None, fx=scale_factor_x, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            # print("close", resized_img.shape, resized_mask.shape)

        else:   # center crop the width to be a multiple of 64
            # print("center crop", end=' - ')
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

            # clip width to be a multiple of 64
            wi_crop = wi_new - (wi_new % 64)
            # center crop image
            resized_img = img[:, (wi_new - wi_crop) // 2 : (wi_new - wi_crop) // 2 + wi_crop, :]
            resized_mask = mask[:, (wi_new - wi_crop) // 2 : (wi_new - wi_crop) // 2 + wi_crop]
            # print("center crop", resized_img.shape, resized_mask.shape)
    
    return resized_img, resized_mask

class Custom_Train_Dataset(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/tp_train.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # source_filename = item['source']
        target_filename = item['target']
        mask_filename = item['mask']
        prompts = item['prompts']
        prompt = np.random.choice(prompts, 1) #, replace=False)


        # source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('../cocoapi/coco/person_new/images/train2017/' + target_filename)
        mask = cv2.imread('../cocoapi/coco/person_new/mask/train2017/' + mask_filename, cv2.IMREAD_GRAYSCALE)
        # print("hi1", type(target), type(prompt))
        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # print(target.shape, mask.shape, "from inside1")
        target, mask = custom_resize(target, mask)
        # print(target.shape, mask.shape, "from inside2\n")
        target = Image.fromarray(target)
        target = self.transform(target)
        target = np.array(target)
        target = target.transpose((1,2,0))
        # print("inside", target)
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32)*2) - 1.0
        # target = target.ToTensor()


        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = plt.imread('../cocoapi/coco/person_new/mask/train2017/' + mask_filename)
        mask = Image.fromarray(mask)
        mask = self.transform(mask)
        mask = np.array(mask)
        mask = (mask>0.5)*1
        mask = mask.transpose((1,2,0))
        # mask = (mask.astype(np.float32)*2) - 1.0

        return dict(jpg=target, mask=mask, txt=str(prompt[0]))

class Custom_Val_Dataset(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/tp_val.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.ToTensor(),           
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # source_filename = item['source']
        target_filename = item['target']
        mask_filename = item['mask']
        prompts = item['prompts']
        prompt =np.random.choice(prompts, 1) #, replace=False)
        target = cv2.imread('../cocoapi/coco/person_new/images/val2017/' + target_filename)
        mask = cv2.imread('../cocoapi/coco/person_new/mask/val2017/' + mask_filename, cv2.IMREAD_GRAYSCALE)
        # print("hi2", type(target), type(prompt))
        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target, mask = custom_resize(target, mask)
        target = Image.fromarray(target)
        target = self.transform(target)
        target = np.array(target)
        target = target.transpose((1,2,0))
        # print("inside", target)
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32)*2) - 1.0
        # target = target.ToTensor()

        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = plt.imread('../cocoapi/coco/person_new/mask/val2017/' + mask_filename)
        # mask = imageio.imread('../cocoapi/coco/person_new/mask/val2017/' + mask_filename)
        mask = Image.fromarray(mask)
        mask = self.transform(mask)
        mask = np.array(mask)
        mask = (mask>0.5)*1
        mask = mask.transpose((1,2,0))
        # mask = (mask.astype(np.float32)*2) - 1.0
        
        return dict(jpg=target, mask=mask, txt=str(prompt[0]))

class Custom_FID_Dataset(Dataset):
    def __init__(self):
        self.data = []
        # randomly read 50 lines from the file
        with open('../cocoapi/PythonAPI/tp_val.json', 'rt') as f:
            # random sample 50 lines
            lines = f.readlines()
            lines = random.sample(lines, 50)
            for line in lines:
                self.data.append(json.loads(line))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),           
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # source_filename = item['source']
        target_filename = item['target']
        mask_filename = item['mask']
        prompts = item['prompts']
        prompt =np.random.choice(prompts, 1) #, replace=False)
        target = cv2.imread('../cocoapi/coco/person_new/images/val2017/' + target_filename)
        mask = cv2.imread('../cocoapi/coco/person_new/mask/val2017/' + mask_filename, cv2.IMREAD_GRAYSCALE)
        # print("hi2", type(target), type(prompt))
        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target, mask = custom_resize(target, mask)
        target = Image.fromarray(target)
        target = self.transform(target)
        target = np.array(target)
        target = target.transpose((1,2,0))
        # print("inside", target)
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32)*2) - 1.0
        # target = target.ToTensor()

        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = plt.imread('../cocoapi/coco/person_new/mask/val2017/' + mask_filename)
        # mask = imageio.imread('../cocoapi/coco/person_new/mask/val2017/' + mask_filename)
        mask = Image.fromarray(mask)
        mask = self.transform(mask)
        mask = np.array(mask)
        mask = (mask>0.5)*1
        mask = mask.transpose((1,2,0))
        # mask = (mask.astype(np.float32)*2) - 1.0
        
        return dict(jpg=target, mask=mask, txt=str(prompt[0]))


class Train_Dataset(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/tp_train.json', 'rt') as f:
            count  = 0
            for line in f:
                self.data.append(json.loads(line))
                count += 1

        self.transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # source_filename = item['source']
        target_filename = item['target']
        prompts = item['prompts']
        prompt =np.random.choice(prompts, 1) #, replace=False)

        # source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('../cocoapi/coco/person/images/train2017/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # target = Image.fromarray(target)
        # target = self.transform(target)
        # target = np.array(target)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=str(prompt))

class Val_Dataset(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/target_prompt_val.json', 'rt') as f:
            count  = 0
            for line in f:
                self.data.append(json.loads(line))
                count += 1
        # self.transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(256),
        #     transforms.ToTensor(),
        # ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # source_filename = item['source']
        target_filename = item['target']
        prompts = item['prompts']
        prompt =np.random.choice(prompts, 1) #, replace=False)
        target = cv2.imread('../cocoapi/coco/person/images/val2017/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # target = Image.fromarray(target)
        # target = self.transform(target)
        # target = np.array(target)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=str(prompt))


class Custom_Train_Dataset_old(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/tp_train.json', 'rt') as f:
            count  = 0
            for line in f:
                self.data.append(json.loads(line))
                count += 1
                if(count == 56000):
                    break
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if idx < 56000:
            item = self.data[idx]

            # source_filename = item['source']
            target_filename = item['target']
            prompts = item['prompts']
            prompt = np.random.choice(prompts, 1) #, replace=False)


            # source = cv2.imread('./training/fill50k/' + source_filename)
            target = cv2.imread('../cocoapi/coco/person/images/train2017/' + target_filename)
            # print("hi1", type(target), type(prompt))
            # Do not forget that OpenCV read images in BGR order.
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            target = Image.fromarray(target)
            target = self.transform(target)
            target = np.array(target)
            target = target.transpose((1,2,0))
            # print("inside", target)
            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32)*2) - 1.0
            # target = target.ToTensor()

        return dict(jpg=target, txt=str(prompt))


class Custom_Val_Dataset_old(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/tp_val.json', 'rt') as f:
            count  = 0
            for line in f:
                self.data.append(json.loads(line))
                count += 1
                if(count==50):
                    break
        # with open('../cocoapi/PythonAPI/target_prompt_training.json', 'rt') as f:
        #     count  = 0
        #     for line in f:
        #         count += 1
        #         if count >= 56000:
        #             self.data.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256), 
            transforms.ToTensor(),           
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # source_filename = item['source']
        target_filename = item['target']
        prompts = item['prompts']
        prompt =np.random.choice(prompts, 1) #, replace=False)
        if idx >=2693:
            target = cv2.imread('../cocoapi/coco/person/images/train2017/' + target_filename)
        else:
            target = cv2.imread('../cocoapi/coco/person/images/val2017/' + target_filename)
        # print("hi2", type(target), type(prompt))
        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = Image.fromarray(target)
        target = self.transform(target)
        target = np.array(target)
        target = target.transpose((1,2,0))
        # print("inside", target)
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32)*2) - 1.0
        # target = target.ToTensor()
        
        return dict(jpg=target, txt=str(prompt))

