"""
Code taken from and modified https://github.com/cydonia999/VGGFace2-pytorch
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import crop
import pickle
__all__ = ['FaceLoss']

MODEL_PATH = "/home/phebbar/Documents/ControlNet/models/resnet50_ft_weight.pkl"


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#loads model params from checkpoint file and ignores fc layer
def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()

    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                    'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            if 'fc' in name:
                pass
            else:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FaceLoss(nn.Module):
    def __init__(self,model_path=None,device="cuda:0"):
        super(FaceLoss, self).__init__()
        MODEL_PATH = model_path
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        self.alphas = [0.1, 0.25 * 0.01, 0.25 * 0.1, 0.25 * 0.2, 0.25 * 0.02]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)

        self.channels = [64, 256, 512, 1024, 2048]
        load_state_dict(self, MODEL_PATH)
        self.to(device=device)

        for param in self.parameters():
            param.requires_grad = False

        self.face_transforms = nn.Sequential(
                                        transforms.Resize(256),
                                        transforms.CenterCrop(254)
                                        )
        self.eval()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward(self, x):  # 224x224
        features = []
        x = self.conv1(x)  # 112x112
        features.append(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56x56 ignore

        x = self.layer1(x)  # 56x56
        features.append(x)
        x = self.layer2(x)  # 28x28
        features.append(x)
        x = self.layer3(x)  # 14x14 ignore (maybe not)
        features.append(x)
        x = self.layer4(x)  # 7x7
        features.append(x)

        return features

    def forward(self, img, rec, bbox):
        """
        Takes in original image and reconstructed image and feeds it through face network and takes the difference
        between the different resolutions and scales by alpha_{i}.
        Normalizing the features and applying spatial resolution was taken from LPIPS and wasn't mentioned in the paper.
        """
        faces = self.prepare_faces(img, rec, bbox)
        if faces is None:
            return img.new_tensor(0)
        faces = faces[:6] # otherwise it may cause oom error.
        features = self._forward(faces)
        features = [f.chunk(2) for f in features]
        # diffs = [a * torch.abs(p[0] - p[1]).sum() for a, p in zip(self.alphas, features)]
        diffs = [a * torch.abs(p[0] - p[1]).sum(dim=0).mean() for a, p in zip(self.alphas, features)]
        # diffs = [a*torch.abs(self.norm_tensor(tf) - self.norm_tensor(rf)) for a, tf, rf in zip(self.alphas, true_features, rec_features)]

        # diffs = [a * torch.mean(torch.abs(tf - rf)) for a, tf, rf in zip(self.alphas, features)]
        return sum(diffs)
        # return sum(diffs) / len(diffs)

    def prepare_faces(self, imgs, recs, bboxes):
        faces_gt = []
        faces_gen = []
        for img, rec, bboxes in zip(imgs, recs, bboxes):
            for bbox in bboxes:
                #looks like image coordinate frame
                top = bbox[1]
                left = bbox[0]
                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]
                crop_img = crop(img, top, left, height, width)
                faces_gt.append(self.face_transforms(crop_img))
                crop_rec = crop(rec, top, left, height, width)
                faces_gen.append(self.face_transforms(crop_rec))
        if len(faces_gt) == 0:
            return None
        faces_gt = torch.stack(faces_gt, dim=0)
        faces_gen = torch.stack(faces_gen, dim=0)
        return torch.cat([faces_gt, faces_gen], dim=0)



if __name__ == '__main__':
    model = FaceLoss()
    # x = torch.randn(1, 3, 256, 256)
    # x_rec = torch.randn(1, 3, 256, 256)
    x = torch.randn(2, 3, 101, 101)
    x_rec = torch.randn(2, 3, 101, 101)
    print(model.forward(x, x_rec))