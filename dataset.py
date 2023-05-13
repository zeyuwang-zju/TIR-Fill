import os, torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def check_image_file(filename):
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])


class TrainDataset(Dataset):
    def __init__(self, image_root, edge_root, mask_root, load_size, crop_size, return_image_root=False):
        super(TrainDataset, self).__init__()

        self.return_image_root = return_image_root
        self.image_files = [os.path.join(root, file) for root, dirs, files in os.walk(image_root)
                            for file in files if check_image_file(file)]
        self.edge_files = [os.path.join(root, file) for root, dirs, files in os.walk(edge_root)
                            for file in files if check_image_file(file)]

        assert len(self.image_files) == len(self.edge_files)
        self.mask_files = [os.path.join(root, file) for root, dirs, files in os.walk(mask_root)
                        for file in files if check_image_file(file)]
        self.number_mask = len(self.mask_files)
        self.image_transforms = transforms.Compose([
                    transforms.Resize(size=load_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomCrop(size=(crop_size, crop_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                ])
        self.mask_transforms = transforms.Compose([
                    transforms.Resize(size=load_size, interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.RandomCrop(size=(crop_size, crop_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor()
                ])
    
    def __getitem__(self, index):
        image = transforms.ToTensor()(Image.open(self.image_files[index]))
        edge = transforms.ToTensor()(Image.open(self.edge_files[index]))
        image, edge = torch.split(self.image_transforms(torch.cat((image, edge), dim=0)), [1, 1], 0)

        mask = Image.open(self.mask_files[random.randint(0, self.number_mask - 1)])
        mask = self.mask_transforms(mask)
        threshold = 0.5
        ones = mask >= threshold
        zeros = mask < threshold
        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)
        mask = 1 - mask

        masked_image = image * mask
        masked_edge = edge * mask

        if not self.return_image_root:
            return masked_image, masked_edge, image, edge, mask
        else:
            return masked_image, masked_edge, image, edge, mask, self.image_files[index]

    def __len__(self):
        return len(self.image_files)