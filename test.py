from dataset import Custom_Train_Dataset, Custom_Val_Dataset, Custom_FID_Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore")
import torch

# seed everything
np.random.seed(42)
torch.manual_seed(42)

# def custom_resize(img, mask):
#     # img shape is (height, width, channels)
#     height, width = img.shape[:2]
#     if(height < width):
#         # transpose image along 0 and 1 axis
#         img, mask = custom_resize(img.transpose(1, 0, 2), mask.transpose(1, 0))
#         # transpose back
#         img = img.transpose(1, 0, 2)
#         mask = mask.transpose(1, 0)
#         return img, mask
#     else:
#         # get the scale factor for height
#         if(height > 256): # resize img if max_dim > 256
#             # Determine the scaling factor to resize the image
#             scale_factor = 256 / height
#         else:
#             # scale such that height is a multiple of 64
#             # get closest multiple of 64
#             dist = np.abs(np.array([64, 128, 192, 256]) - height)
#             closest_multiple = (1+np.argmin(dist))*64
#             scale_factor = closest_multiple / height
#             # scale_factor = max( 64 / height, 128/height, 256/height)
#         # Resize the image
#         wi_new = int(width * scale_factor)
#         hi_new = int(height * scale_factor)
#         # print("raw scale", hi_new, wi_new, end=' - ')
#         if(wi_new<64):
#             # resize width to 64
#             scale_factor_x = 64 / width
#             resized_img = cv2.resize(img, None, fx=scale_factor_x, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
#             resized_mask = cv2.resize(mask, None, fx=scale_factor_x, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

#         elif(wi_new%64 < 10 or wi_new%64 > 54): #the width is very near to a multiple of 64, so just resize
#             # print("close", end=' - ')
#             # get nearest multiple of 64
#             dist = np.abs(np.array([64, 128, 192, 256]) - wi_new)
#             closest_multiple = (1+np.argmin(dist))*64
#             scale_factor_x = closest_multiple / width
#             resized_img = cv2.resize(img, None, fx=scale_factor_x, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
#             resized_mask = cv2.resize(mask, None, fx=scale_factor_x, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

#         else:   # center crop the width to be a multiple of 64
#             print("center crop", end=' - ')
#             img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

#             # clip width to be a multiple of 64
#             wi_crop = wi_new - (wi_new % 64)
#             # center crop image
#             resized_img = img[:, (wi_new - wi_crop) // 2 : (wi_new - wi_crop) // 2 + wi_crop, :]
#             resized_mask = mask[:, (wi_new - wi_crop) // 2 : (wi_new - wi_crop) // 2 + wi_crop]
    
#     return resized_img, resized_mask

# test the custom_resize function
# generate a random image
# img = cv2.imread('test_imgs/bag_scribble.png')
# mask = cv2.imread('test_imgs/bag_scribble.png', cv2.IMREAD_GRAYSCALE)
# # extend image to hxwx1
# # print(img.shape)
# for i in range(50):
#     height = np.random.randint(50, 500)
#     width = np.random.randint(50, 500)
#     height = 300
#     width = 61
#     # resize image to 256 193
#     img = cv2.resize(img, (height, width))
#     mask = cv2.resize(mask, (height, width))
#     print("(", img.shape, mask.shape, ")", end=' - ')
#     resized_img, resized_mask = custom_resize(img, mask)
#     print("(", resized_img.shape, resized_mask.shape, ")")
# # save images
# cv2.imwrite('original.png', img)
# # cv2.imwrite('resized.png', resized_img)

dataset = Custom_Train_Dataset()
# dataset = Custom_Val_Dataset()
# dataset = Custom_FID_Dataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
for batchid, batch in enumerate(dataloader):
    print(batchid, batch["jpg"].shape, batch["mask"].shape, batch["face_boxes"].shape)
    img = batch["jpg"][0].numpy()
    # print(img.shape)
    # face_box = batch["face_boxes"]
    # print(face_box.shape)
    # if(face_box.shape[2]!=4):
    #     print("face box shape is not 4")
    #     break
    # print(batchid, face_box.shape)
    # print(face_box.shape)
    # face_box = face_box[0].numpy()
    # face_box = np.array([[10, 30, 40, 70]])
    # show face boxes on image
    # if(batchid == 40):
    #     for i in range(face_box.shape[0]):
    #         cv2.rectangle(img, (face_box[i, 0], face_box[i, 1]), (face_box[i, 2], face_box[i, 3]), (0, 255, 0), 2)
    #     # show image
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    # print(batch["jpg"].shape, batch["mask"].shape)
