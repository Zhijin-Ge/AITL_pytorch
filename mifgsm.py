import torch
import random
from utils import *
from attack import Attack
import torch.nn.functional as F

def input_admix(input_tensor, portion=0.2):
    shuffle_index = torch.randperm(input_tensor.shape[0])
    return torch.clamp(input_tensor + portion*input_tensor[shuffle_index], 0.0, 1.0)

def input_admix_and_scale(input_tensor, portion=0.2):
    shuffle_index = torch.randperm(input_tensor.shape[0])
    scale_factor = random.randint(0, 4)
    return torch.clamp((input_tensor + portion*input_tensor[shuffle_index]) / (2 ** scale_factor), 0.0, 1.0)

def input_scale(input_tensor):
    scale_factor = random.randint(0, 4)
    return input_tensor / (2 ** scale_factor)

def blend(image_1, image_2, factor):
    processed_image = image_1 * (1 - factor) + image_2 * factor
    return torch.clamp(processed_image, 0, 1.0)

def input_brightness(input_tensor, factor_delta=0.5):
    factor = torch.FloatTensor(1).uniform_(1 - factor_delta, 1 + factor_delta)[0]
    degenerate = torch.zeros_like(input_tensor)
    processed_image = blend(degenerate, input_tensor, factor)
    return processed_image

def input_color(input_tensor, factor_delta=0.5):
    factor = torch.FloatTensor(1).uniform_(1 - factor_delta, 1 + factor_delta)[0]
    degenerate = 0.2989 * input_tensor[:, 0, :, :] + 0.5870 * input_tensor[:, 1, :, :] + 0.1140 * input_tensor[:, 2, :, :]
    degenerate = degenerate.unsqueeze(1).expand_as(input_tensor)
    processed_image = blend(degenerate, input_tensor, factor)
    return processed_image

def input_contrast(input_tensor, factor_delta=0.5):
    factor = torch.FloatTensor(1).uniform_(1 - factor_delta, 1 + factor_delta)[0]
    degenerate = 0.2989 * input_tensor[:, 0, :, :] + 0.5870 * input_tensor[:, 1, :, :] + 0.1140 * input_tensor[:, 2, :, :]
    mean = degenerate.mean(dim=[1, 2], keepdim=True)
    mean = mean.unsqueeze(1)
    degenerate = torch.ones_like(input_tensor) * mean
    processed_image = blend(degenerate, input_tensor, factor)
    return processed_image

def get_sharpness_kernel():
    sharpness_kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32)
    sharpness_kernel = sharpness_kernel / np.sum(sharpness_kernel)
    sharpness_kernel = np.stack([sharpness_kernel, sharpness_kernel, sharpness_kernel])
    sharpness_kernel = np.expand_dims(sharpness_kernel, 0)
    return torch.from_numpy(sharpness_kernel).cuda()

def input_sharpness(input_tensor, factor_delta=0.5):
    sharpness_kernel = get_sharpness_kernel()
    factor = torch.FloatTensor(1).uniform_(1 - factor_delta, 1 + factor_delta)[0]
    degenerate = F.conv2d(input_tensor, sharpness_kernel, bias=None, stride=(1, 1), padding=1)
    processed_image = blend(degenerate, input_tensor, factor)
    return processed_image


def input_shearX(input_tensor, delta=0.5):
    factor = torch.FloatTensor(1).uniform_(-delta, delta)[0]
    N, C, W, H = input_tensor.size()
    matrix = torch.tensor([1, factor, 0, 0, 1, 0]).view(2, 3).repeat(N, 1, 1).cuda()
    size = torch.Size((N, C, W, H))
    grid = F.affine_grid(matrix, size)
    processed_image = F.grid_sample(input_tensor, grid)
    return processed_image

def input_shearY(input_tensor, delta=0.5):
    factor = torch.FloatTensor(1).uniform_(-delta, delta)[0]
    N, C, W, H = input_tensor.size()
    matrix = torch.tensor([1, 0, 0, factor, 1, 0]).view(2, 3).repeat(N, 1, 1).cuda()
    size = torch.Size((N, C, W, H))
    grid = F.affine_grid(matrix, size)
    processed_image = F.grid_sample(input_tensor, grid)
    return processed_image


def input_translateX(input_tensor, delta=0.4):
    factor = torch.FloatTensor(1).uniform_(-delta, delta)[0]
    N, C, W, H = input_tensor.size()
    matrix = torch.tensor([1, 0, factor, 0, 1, 0]).view(2, 3).repeat(N, 1, 1).cuda()
    size = torch.Size((N, C, W, H))
    grid = F.affine_grid(matrix, size)
    processed_image = F.grid_sample(input_tensor, grid)
    return processed_image


def input_translateY(input_tensor, delta=0.4):
    factor = torch.FloatTensor(1).uniform_(-delta, delta)[0]
    N, C, W, H = input_tensor.size()
    matrix = torch.tensor([1, 0, 0, 0, 1, factor]).view(2, 3).repeat(N, 1, 1).cuda()
    size = torch.Size((N, C, W, H))
    grid = F.affine_grid(matrix, size)
    processed_image = F.grid_sample(input_tensor, grid)
    return processed_image


def input_reshape(input_tensor, delta=0.5):
    N, C, W, H = input_tensor.size()
    scale_x = torch.FloatTensor(1).uniform_(1 - delta, 1 + delta)[0]
    scale_y = torch.FloatTensor(1).uniform_(1 - delta, 1 + delta)[0]
    shear_x = torch.FloatTensor(1).uniform_(-delta, delta)[0]
    shear_y = torch.FloatTensor(1).uniform_(-delta, delta)[0]
    translate_x = torch.FloatTensor(1).uniform_(-delta, delta)[0]
    translate_y = torch.FloatTensor(1).uniform_(-delta, delta)[0]
    matrix = torch.tensor([scale_x, shear_x, translate_x, shear_y, scale_y, translate_y]).view(2, 3).repeat(N, 1, 1).cuda()
    size = torch.Size((N, C, W, H))
    grid = F.affine_grid(matrix, size)
    processed_image = F.grid_sample(input_tensor, grid)
    return processed_image


def input_rotate(input_tensor, theta=np.pi/6):
    transform = transforms.RandomRotation(degrees=[-theta, theta])
    processed_image = transform(input_tensor)
    return processed_image


def input_crop(input_tensor):
    _, _, image_width, image_height = input_tensor.shape
    rnd = torch.randint(279, image_width, ())

    croped = F.interpolate(input_tensor, size=(rnd, rnd), mode='bilinear')

    h_rem = image_height - rnd
    w_rem = image_width - rnd

    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left

    padded = F.pad(croped, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.)
    processed_image = F.interpolate(padded, size=(image_height, image_width), mode='bilinear')
    return processed_image

def input_resize(input_tensor, resize_rate=1.15, diversity_prob=1.0):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = input_tensor.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    processed_image = padded if torch.rand(1) < diversity_prob else input_tensor
    return processed_image

def input_gamma(input_tensor, delta=1.0):
    random_delta = torch.FloatTensor(input_tensor.size(0), 1, 1, 1).uniform_(1 - delta, 1 + delta).cuda()
    processed_image = (input_tensor) + 1e-10
    processed_image = torch.pow(processed_image, random_delta)
    processed_image = torch.clamp(processed_image, 0.0, 1.0)
    processed_image = processed_image
    return processed_image

def input_cutout(input_tensor):
    transform = transforms.RandomErasing(p=1, scale=(0.04, 0.04), ratio=(0.4, 0.4), value=(0, 0, 0))
    processed_image = transform(input_tensor).cuda()
    return processed_image

def input_invert(input_tensor): # 取反
    processed_image = -(input_tensor-0.5) + 0.5
    processed_image = -input_tensor.cuda()
    return processed_image

def rgb_to_hsv(img):
    eps = 1e-6
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)
    hue[img[:,2]==img.max(1)[0]] = 4.0 + ((img[:,0]-img[:,1]) / (img.max(1)[0] - img.min(1)[0] + eps))[img[:,2]==img.max(1)[0]]
    hue[img[:,1]==img.max(1)[0]] = 2.0 + ((img[:,2]-img[:,0]) / (img.max(1)[0] - img.min(1)[0] + eps))[img[:,1]==img.max(1)[0]]
    hue[img[:,0]==img.max(1)[0]] = (0.0 + ((img[:,1]-img[:,2]) / (img.max(1)[0] - img.min(1)[0] + eps))[img[:,0]==img.max(1)[0]]) % 6

    hue[img.min(1)[0]==img.max(1)[0]] = 0.0
    hue = hue/6

    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0]==0] = 0

    value = img.max(1)[0]
        
    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value],dim=1)
    return hsv

def hsv_to_rgb(hsv):
    h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
    h = h%1
    s = torch.clamp(s,0,1)
    v = torch.clamp(v,0,1)
  
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
        
    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))
        
    hi0 = hi==0
    hi1 = hi==1
    hi2 = hi==2
    hi3 = hi==3
    hi4 = hi==4
    hi5 = hi==5
        
    r[hi0] = v[hi0]
    g[hi0] = t[hi0]
    b[hi0] = p[hi0]
        
    r[hi1] = q[hi1]
    g[hi1] = v[hi1]
    b[hi1] = p[hi1]
        
    r[hi2] = p[hi2]
    g[hi2] = v[hi2]
    b[hi2] = t[hi2]
        
    r[hi3] = p[hi3]
    g[hi3] = q[hi3]
    b[hi3] = v[hi3]
        
    r[hi4] = t[hi4]
    g[hi4] = p[hi4]
    b[hi4] = v[hi4]
        
    r[hi5] = v[hi5]
    g[hi5] = p[hi5]
    b[hi5] = q[hi5]
        
    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)
    return rgb

def input_hue(input_tensor, delta=0.2): # 色相
    N, C, W, H = input_tensor.size()
    random_delta = torch.FloatTensor(N, 1, 1).uniform_(-delta, delta)
    processed_image = rgb_to_hsv(input_tensor)
    mask_shape = [N, W, H]
    mask = torch.ones(mask_shape) * random_delta
    mask = torch.stack([mask, torch.zeros(mask_shape), torch.zeros(mask_shape)], dim=1)
    processed_image = processed_image + mask.cuda()
    processed_image = torch.clamp(processed_image, 0.0, 1.0)
    processed_image = hsv_to_rgb(processed_image)
    processed_image = processed_image
    return processed_image

def input_saturation(input_tensor): # 饱和度
    transform = transforms.ColorJitter(saturation=10)
    processed_image = transform(input_tensor).cuda()
    return processed_image


def transform_index(data, trans_index, **kwargs):
    if trans_index==1:
        return input_admix(data)
    elif trans_index==2:
        return input_admix_and_scale(data)
    elif trans_index==3:
        return input_scale(data)
    elif trans_index==4:
        return input_brightness(data)
    elif trans_index==5:
        return input_color(data)
    elif trans_index==6:
        return input_contrast(data)
    elif trans_index==7:
        return input_sharpness(data)
    elif trans_index==8:
        return input_shearX(data)
    elif trans_index==9:
        return input_shearY(data)
    elif trans_index==10:
        return input_translateX(data)
    elif trans_index==11:
        return input_translateY(data)
    elif trans_index==12:
        return input_reshape(data)
    elif trans_index==13:
        return input_rotate(data)
    elif trans_index==14:
        return input_crop(data)
    elif trans_index==15:
        return input_resize(data)
    elif trans_index==16:
        return input_gamma(data)
    elif trans_index==17:
        return input_cutout(data)
    elif trans_index==18:
        return input_invert(data)
    elif trans_index==19:
        return input_hue(data)
    elif trans_index==20:
        return input_saturation(data)
    else:
        print('Index out! Range is [1, 20]')
        return
    
class MIFGSM(Attack):
    """
    MI-FGSM Attack
    'Boosting Adversarial Attacks with Momentum (CVPR 2018)'(https://arxiv.org/abs/1710.06081)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
    """
    
    def __init__(self, model, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., transform_list=[1, 2, 3, 4], targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='MI-FGSM', **kwargs):
        super().__init__(attack, model, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.transform_list = transform_list
    
    def transform(self, data, **kwargs):
        for trans_index in self.transform_list:
            data = transform_index(data, trans_index)
        return data