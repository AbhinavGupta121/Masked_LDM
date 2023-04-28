import torch
import lpips
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("cuda:0")
# LPIPS needs the images to be in the [-1, 1] range.

loss_fn = lpips.LPIPS(net='vgg',verbose=False).to("cuda:0")
# loss_mask = loss_fn.forward(x0_pred_img*mask, x0_gt*mask).mean()

img1 ="image_log/version217/train/loss/loss_gs-000000_e-000000_b-000000.png"
img1 = Image.open(img1)
img1 = np.array(img1)

img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
img1 /= 255.0

noise_img1 = img1[:,:,2:514,2:514]
predx0_img1 = img1[:,:,2:514,516:1028]
gt_img1 = img1[:,:,2:514,1030:1542]
mask_img1 = img1[:,:,2:514,1544:2056]

noise_img1=noise_img1.to("cuda:0")
gt_img1=gt_img1.to("cuda:0") # is in the range 0-1
predx0_img1=predx0_img1.to("cuda:0")
mask_img1=mask_img1.to("cuda:0")


custom_mask1 = torch.zeros((512,512)).to("cuda:0")
custom_mask1[0:256,:]=1
num_pixels1 = torch.sum(custom_mask1)
print("num_pixels1", num_pixels1)

custom_mask2 = torch.zeros((512,512)).to("cuda:0")
custom_mask2[0:25,:]=1
num_pixels2 = torch.sum(custom_mask2)
print("num_pixels2", num_pixels2)

print("with mask1", loss_fn.forward(gt_img1*custom_mask1, predx0_img1*custom_mask1, retPerLayer=False, normalize=True).mean()*512*512/num_pixels1)
print("with mask2", loss_fn.forward(gt_img1*custom_mask2, predx0_img1*custom_mask2, retPerLayer=False, normalize=True).mean()*512*512/num_pixels2)


# mask_img1 = (mask_img1 > 0.5)
# num_pixels = torch.sum(mask_img1)/3
# print("number of pixels in mask out of 512*512:", num_pixels.sum())#58112



# print("without mask")#0.3278
# # print("same images ka lpips:", loss_fn.forward(gt_img1, gt_img1, retPerLayer=False, normalize=True).mean())
# print("gt aur pred images ka lpips:", loss_fn.forward(gt_img1, predx0_img1, retPerLayer=False, normalize=True).mean())

# print("with mask")#0.2576
# # print("same images ka lpips:", loss_fn.forward(gt_img1*mask_img1, gt_img1*mask_img1, retPerLayer=False, normalize=True).mean()*512*512/num_pixels)
# print("gt aur pred images ka lpips:", loss_fn.forward(gt_img1*mask_img1, predx0_img1*mask_img1, retPerLayer=False, normalize=True).mean()*512*512/num_pixels)


# # to check if area change krne pe loss change ho rha hai ki nhi

# print("100*100 vala area") #=0.2383
# print("gt aur pred images ka lpips:", loss_fn.forward(gt_img1[:,:,250:350,250:350], predx0_img1[:,:,250:350,250:350], retPerLayer=False, normalize=True).mean())

# print("150*150 vala area")#=0.2224
# print("gt aur pred images ka lpips:", loss_fn.forward(gt_img1[:,:,225:375,225:375], predx0_img1[:,:,225:375,225:375], retPerLayer=False, normalize=True).mean())









# print("same images ka lpips:", lpips(gt_img1, gt_img1))
# print("gt aur pred images ka lpips:", lpips(gt_img1, predx0_img1))
# print("gt aur mask images ka lpips:", lpips(gt_img1, mask_img1))

# img1 = (torch.rand(1, 3, 100, 100) * 2) - 1
# img2 = (torch.rand(1, 3, 100, 100) * 2) - 1
# img1=img1.to("cuda:0")
# img2=img2.to("cuda:0")
# print(lpips(img1, img2))

# loss_fn = lpips.LPIPS(net='vgg',verbose=False).to(x0_gt.device)
# x0_pred = self.get_x0(x_noisy, t, cond, model_output)
# x0_pred_img = self.decode_first_stage(x0_pred)
# x0_gt = x0_gt.permute(0, 3, 1, 2)
# mask = mask.permute(0, 3, 1, 2)
# # print devices
# # print(x0_pred_img.device, x0_gt.device, mask.device)
# # print(x0_pred_img.shape, x0_gt.shape, mask.shape)
# loss_mask = loss_fn.forward(x0_pred_img*mask, x0_gt*mask).mean()