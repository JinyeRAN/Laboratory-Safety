import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import PIL
import numpy as np

from PIL import Image
from torchvision import models
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize

import warnings
warnings.filterwarnings('ignore')


def label_colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

def label2rgb(lbl, img=None, n_labels=None, alpha=0.3):
    if n_labels is None:
        n_labels = len(np.unique(lbl))
    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)
    cmap[1,0] = 255.0
    try :
        cmap[2,1] = 255.0
    except IndexError as e:
        pass
    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        img_gray = PIL.Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz

class Transform_test(object):
    def __init__(self,size):
        self.size = size
    def __call__(self, input, target):
        # do something to both images
        input =  input.resize(self.size, Image.BILINEAR)
        target = target.resize(self.size,Image.NEAREST)

        target = torch.from_numpy(np.array(target)).long().unsqueeze(0)
        input_tensor = ToTensor()(input)
        Normalize([.485, .456, .406], [.229, .224, .225])(input_tensor)
        return input_tensor, target, input

class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class SegNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        decoders = list(models.vgg16(pretrained=True).features.children())

        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
        self.dec5 = nn.Sequential(*decoders[24:])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad = False

        self.enc5 = SegNetEnc(512, 512, 1)
        self.enc4 = SegNetEnc(1024, 256, 1)
        self.enc3 = SegNetEnc(512, 128, 1)
        self.enc2 = SegNetEnc(256, 64, 0)
        self.enc1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)
        enc5 = self.enc5(dec5)

        enc4 = self.enc4(torch.cat([dec4, enc5], 1))
        enc3 = self.enc3(torch.cat([dec3, enc4], 1))
        enc2 = self.enc2(torch.cat([dec2, enc3], 1))
        enc1 = self.enc1(torch.cat([dec1, enc2], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])


def main():
    mask_root = './data/liquid/mask.png'
    mask = Image.open(mask_root).convert('L')
    mask = mask.resize((672, 480), Image.NEAREST)
    mask = torch.from_numpy(np.array(mask)).long().unsqueeze(0)

    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    model = SegNet(2).cuda()
    model.load_state_dict(torch.load('./weights/liquid_weight.pth'))
    model.eval()

    root_image = './dataset/liquid/JPEGImages'
    root_label = './dataset/liquid/gtFine'

    for filename in os.listdir(root_image):
        img = os.path.join(root_image, filename)
        lbl = os.path.join(root_label, filename)

        image = Image.open(img).convert('RGB')
        label = Image.open(lbl).convert('P')

        image_tensor, label_tensor, img = Transform_test((672, 480))(image, label)
        outputs = model(image_tensor.unsqueeze(0).cuda())
        out = outputs[0].cpu().max(0)[1].data.squeeze(0).byte().numpy()
        mask = mask.squeeze(0).squeeze(0)
        out[mask == 0] = 0
        # label2img = label2rgb(out, np.array(img), n_labels=2)
        # Image.fromarray(label2img).save('./results/' + filename)

        ratio = 1 - out.sum() / mask.sum().item()
        if ratio>0.95:
            print('Full')


if __name__ == '__main__':
    main()


