import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from mmcv.runner import force_fp32, auto_fp16

"""GridMask是一种数据增强技术，可以用于深度学习训练中，以增加模型的泛化能力。
它通过在输入图像上添加一些网格形状的遮罩，使模型在训练时无法看到图像的部分区域，
从而迫使模型学习到更多的上下文信息。

Grid类的主要方法是__call__，它接受一个图像和一个标签，然后在图像上添加一个网格遮罩。
遮罩的大小、位置和旋转角度都是随机生成的。

GridMask类是一个PyTorch模块，可以直接作为一个层添加到神经网络中。
它的主要方法是forward，它接受一个输入张量，然后在张量的每个通道上添加一个网格遮罩。

Grid和GridMask类都有一个set_prob方法，可以用来动态调整遮罩的概率。
这是一个常见的技术，可以在训练初期使用更少的遮罩，然后随着训练的进行逐渐增加遮罩的数量，使模型逐渐适应更复杂的输入。
"""

class Grid(object):

    def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode=mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1-mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask 

        return img, label


class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch # prob随着训练的进行而增大
    @auto_fp16()
    def forward(self, x):  # x:[14,3,374,640]
        """
            计算遮挡的高度和宽度，以及遮挡的大小d。
            创建一个全为1的遮挡矩阵mask。
            计算遮挡的起始位置st_h和st_w。
            如果设置了使用高度遮挡，则在遮挡矩阵mask上，将每个高度区间的一部分设置为0，表示遮挡。
            如果设置了使用宽度遮挡，则在遮挡矩阵mask上，将每个宽度区间的一部分设置为0，表示遮挡。
            随机旋转遮挡矩阵。
            将遮挡矩阵裁剪到与输入图像相同的大小，并转为torch张量，放到GPU上。
            如果设置了模式1，则遮挡矩阵取反。
            将遮挡矩阵扩展为与输入相同的尺寸。
            如果设置了偏移，则输入图像在遮挡部分加上随机偏移，否则输入图像直接乘以遮挡矩阵，实现遮挡效果。
            最后返回遮挡后的图像。
        """
        if np.random.rand() > self.prob or not self.training:
            return x
        n,c,h,w = x.size()
        x = x.view(-1,h,w)
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).to(x.dtype).cuda()
        if self.mode == 1:
            mask = 1-mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).to(x.dtype).cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask 
        
        return x.view(n,c,h,w)