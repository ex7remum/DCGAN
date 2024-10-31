# adapted from https://github.com/photosynthesis-team/piq/discussions/319#discussioncomment-2976851

from torch.utils.data import Dataset
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

def getFilePaths(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    ret.sort()
    return ret

class DataProcess(Dataset):
    def __init__(self, s_src, img_w=64, img_h=64, limit = 1000):
        super(DataProcess, self).__init__()
        self.img_w = img_w
        self.img_h = img_h

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_h, self.img_w)),
            transforms.ToTensor(),
        ])

        self.f_srcs = getFilePaths(s_src)[:limit]

    def __getitem__(self, index):
        src = Image.open(self.f_srcs[index])
        t_src = self.img_transform(src.convert('RGB'))

        return {
            'images': t_src,
        }

    def __len__(self):
        return len(self.f_srcs)
