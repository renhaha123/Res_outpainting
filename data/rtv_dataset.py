import torch
from torch.utils.data import Dataset
import glob
from torchvision.transforms import transforms
from PIL import Image
import random
import os


class Places2(Dataset):
    def __init__(self, opt):
        super(Places2, self).__init__()
        self.opt = opt
        self.rtv_img_file = sorted(glob.glob('{:s}/*'.format(opt.rtv_img_file)))
        if opt.mask_file:
            self.mask_file = sorted(glob.glob('{:s}/*'.format(opt.mask_file)))

        self.img_transform = transforms.Compose([
            transforms.Resize(opt.fine_size, opt.fine_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(opt.fine_size, opt.fine_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):

        rtv_img = Image.open(self.rtv_img_file[item]).convert('RGB')
        rtv_img = self.img_transform(rtv_img)

        if self.opt.mask_file:
            if self.opt.train:
                index = random.randint(0, len(self.mask_file)-1)
            else:
                index = item
            mask = Image.open(self.mask_file[index]).comvert('L')
            mask = self.mask_transform(mask)
        else:
            mask = load_center_mask(rtv_img)

        img_name = os.path.basename(self.rtv_img_file[item])

        return rtv_img, mask, img_name

    def __len__(self):
        return len(self.rtv_img_file)


def load_center_mask(img):
    _, h, w = img.size()
    mask = torch.ones(1, h, w)
    mask[:, int(h/4):int(-h/4), int(w/4):int(-w/4)] = 0
    mask = 1 - mask
    return mask