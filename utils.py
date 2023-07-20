import pandas as pd
from PIL import Image
import os
import time
import sys
import random
from torchvision import transforms
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.models as models
import timm

img_max, img_min = 1., 0

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def get_model(net_name):
    """Load converted model"""
    if net_name == 'resnet18':
        model = wrap_model(models.resnet18(pretrained=True).eval().cuda())
    elif net_name == 'resnet50':
        model = wrap_model(models.resnet50(pretrained=True).eval().cuda())
    elif net_name == 'resnet101':
        model = wrap_model(models.resnet101(pretrained=True).eval().cuda())
    elif net_name == 'resnet152':
        model = wrap_model(models.resnet101(pretrained=True).eval().cuda())
    elif net_name == 'densenet201':
        model = wrap_model(models.densenet201(pretrained=True).eval().cuda())
    elif net_name == 'mobilenetv2':
        model = wrap_model(models.mobilenet_v2(pretrained=True).eval().cuda())
    elif net_name == 'mobilenetv3':
        model = wrap_model(models.mobilenet_v3_large(pretrained=True).eval().cuda())
    elif net_name == 'vgg19':
        model = wrap_model(models.vgg19(pretrained=True).eval().cuda())
    elif net_name == 'efficientnet_b4':
        model = wrap_model(models.efficientnet_b4(pretrained=True).eval().cuda())
    elif net_name == 'efficientnetv2_s':
        model = wrap_model(models.efficientnet_v2_s(pretrained=True).eval().cuda())
    elif net_name == 'xception65':
        model = wrap_model(timm.create_model(net_name, pretrained=True).eval().cuda())
    elif net_name == 'senet154':
        model = wrap_model(timm.create_model(net_name, pretrained=True).eval().cuda())
    elif net_name == 'adv_inception_v3':
        model = wrap_model(timm.create_model(net_name, pretrained=True).eval().cuda())
    elif net_name == 'Incv3':
        model = wrap_model(models.inception_v3(pretrained=True).eval().cuda())
    elif net_name == 'Incv4':
        model = wrap_model(models.inception_v4(pretrained=True).eval().cuda())
    elif net_name == 'IncResv2':
        model = wrap_model(models.inceptionresnetv2(pretrained=True).eval().cuda())
    else:
        print('Wrong model name!')
    return model

def wrap_model(model):
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)

def save_images(output_dir, adversaries, filenames):
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)

def load_labels(untarget, dataset='imagenet'):
    if dataset == 'imagenet':
        file_name = './data/dev_dataset.csv'
        dev = pd.read_csv(file_name)
        label = 'TrueLabel' if untarget else 'TargetClass'
        f2l = {dev.iloc[i]['ImageId']+'.png': dev.iloc[i][label] for i in range(len(dev))}
    elif dataset == 'imagenet_valrs':
        file_name = './data/val_rs.csv'
        dev = pd.read_csv(file_name)
        assert untarget is True
        label = 'label'
        f2l = {dev.iloc[i]['filename']: dev.iloc[i][label] for i in range(len(dev))}
    return f2l


class load_images(data.Dataset):
    def __init__(self, dir, input_csv):
        self.dir = dir
        self.csv = pd.read_csv(input_csv)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(299)
        ])

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['filename']
        Truelabel = img_obj['label']
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        return ImageID, data, Truelabel

    def __len__(self):
        return len(self.csv)


if __name__=='__main__':
    input_dir = './data/val_rs'
    input_csv = './data/val_rs.csv'
    batch_size = 10
    noise = 4
    X = load_images(input_dir, input_csv)
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    for images_ID, images, true_label in tqdm(data_loader):
        # print(images_ID)
        print(images.shape)
        print(true_label.shape)
        save_images('./outputs', images, images_ID)
