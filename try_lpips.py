import torch
_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("cuda:0")
# LPIPS needs the images to be in the [-1, 1] range.
img1 = (torch.rand(10, 3, 100, 100) * 2) - 1
img2 = (torch.rand(10, 3, 100, 100) * 2) - 1
img1=img1.to("cuda:0")
img2=img2.to("cuda:0")
print(lpips(img1, img2))

#print