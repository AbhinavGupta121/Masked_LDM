import json
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from torch.utils.data import Dataset


def custom_resize(img):
    height, width = img.shape[:2]

    # If the image is already smaller than or equal to 256, return it as is
    if height <= 256 and width <= 256:
        return img

    # Determine the scaling factor to resize the image
    scale_factor = min(256 / height, 256 / width)

    # Resize the image
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Return the resized image
    return resized_img

class Custom_Train_Dataset(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/tp_train.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.transform = transforms.Compose([
            # transforms.Resize(256, max_size=256),
            # transforms.CenterCrop(256),
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
        # print("hi1", type(target), type(prompt))
        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = custom_resize(target)
        target = Image.fromarray(target)
        target = self.transform(target)
        target = np.array(target)
        target = target.transpose((1,2,0))
        # print("inside", target)
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32)*2) - 1.0
        # target = target.ToTensor()


        mask = cv2.imread('../cocoapi/coco/person_new/mask/train2017/' + mask_filename, cv2.IMREAD_GRAYSCALE)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = plt.imread('../cocoapi/coco/person_new/mask/train2017/' + mask_filename)
        mask = imageio.imread('../cocoapi/coco/person_new/mask/train2017/' + mask_filename)
        mask = custom_resize(mask)
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
            # transforms.Resize(256, max_size=256),
            # transforms.CenterCrop(256), 
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
        # print("hi2", type(target), type(prompt))
        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = custom_resize(target)
        target = Image.fromarray(target)
        target = self.transform(target)
        target = np.array(target)
        target = target.transpose((1,2,0))
        # print("inside", target)
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32)*2) - 1.0
        # target = target.ToTensor()

        mask = cv2.imread('../cocoapi/coco/person_new/mask/val2017/' + mask_filename, cv2.IMREAD_GRAYSCALE)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # mask = plt.imread('../cocoapi/coco/person_new/mask/val2017/' + mask_filename)
        # mask = imageio.imread('../cocoapi/coco/person_new/mask/val2017/' + mask_filename)
        mask = custom_resize(mask)
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

