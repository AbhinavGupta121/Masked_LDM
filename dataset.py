import json
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import Dataset

class Train_Dataset(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/target_prompt_training.json', 'rt') as f:
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


class Custom_Train_Dataset(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/target_prompt_training.json', 'rt') as f:
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


class Custom_Val_Dataset(Dataset):
    def __init__(self):
        self.data = []
        with open('../cocoapi/PythonAPI/target_prompt_val.json', 'rt') as f:
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

