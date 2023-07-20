"""Implementation of evaluate attack result."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
# from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torchvision.models as models
import timm
from utils import *

# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

batch_size = 10

input_csv = './data/val_rs.csv'
input_dir = './path/to/data/images'
adv_dir = './results/mifgsm'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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


def get_vit_model(net_name):
    """Load converted model"""
    model = wrap_model(timm.create_model(net_name, pretrained=True).eval().cuda())

    return model


def main():
    res = '|'
    my_model = ['mobilenetv2', 'mobilenetv3', 'resnet50', 'resnet101', 'densenet201', 'vgg19', 'xception65', 'resnet152', 'efficientnet_b4', 'efficientnetv2_s', 'adv_inception_v3']

    X = load_images(adv_dir, input_csv)
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    # for model_name in Cnn_model_names:
    #     model = get_model(model_name)
    #     sum_acc = 0
    #     for _, images, gt_cpu in data_loader:
    #         gt = gt_cpu.cuda()
    #         images = images.cuda()
    #         with torch.no_grad():
    #             sum_acc += (model(images).argmax(1)!=(gt)).detach().sum().cpu()
    #     print(model_name, sum_acc)
    #     res += f'{model_name}:' + 'acu = {:.2%} |'.format(sum_acc / 1000.0)

    for model_name in my_model:
        model = get_model(model_name)
        sum_acc = 0
        for _, images, gt_cpu in data_loader:
            gt = gt_cpu.cuda()
            images = images.cuda()
            with torch.no_grad():
                sum_acc += (model(images).argmax(1)!=(gt)).detach().sum().cpu()
        print(model_name, sum_acc)
        res += f'{model_name}:' + 'acu = {:.2%} |'.format(sum_acc / 1000.0)

    with open('results_eval.txt', 'a') as f:
        f.write(adv_dir + res + '\n')

if __name__ == '__main__':
    print(adv_dir)
    main()