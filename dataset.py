import json
import cv2
import numpy as np

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/coco_person/target_prompt.json', 'rt') as f:
            count  = 0
            for line in f:
                self.data.append(json.loads(line))
                count += 1
                # if(count == 1000):
                #     break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # source_filename = item['source']
        target_filename = item['target']
        prompts = item['prompts']
        prompt =np.random.choice(prompts, 1, replace=False)


        # source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/coco_person/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # return dict(jpg=target, txt=prompt, hint=source)
        return dict(jpg=target, txt=prompt)

