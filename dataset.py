import os, torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import Normalize


def check_image_file(filename):
    """
    用于判断filename是否为图片
    """
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])


class TrainDataset(Dataset):
    def __init__(self, image_root, edge_root, mask_root, load_size, crop_size, return_image_root=False):
        super(TrainDataset, self).__init__()
        """
        image_root: 存放数据集图片的地址
        mask_root: 存放mask图片的地址
        load_size: 读取图片后resize成的大小
        crop_size: 图片经resize之后随机裁剪出的大小
        """
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
                    # transforms.ToTensor(),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        self.mask_transforms = transforms.Compose([
                    transforms.Resize(size=load_size, interpolation=transforms.InterpolationMode.NEAREST), # 因为mask只有0和1两种数值，因此用nearestr
                    transforms.RandomCrop(size=(crop_size, crop_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor()
                ])
    
    def __getitem__(self, index):
        image = transforms.ToTensor()(Image.open(self.image_files[index]))
        edge = transforms.ToTensor()(Image.open(self.edge_files[index]))
        image, edge = torch.split(self.image_transforms(torch.cat((image, edge), dim=0)), [1, 1], 0)
        # print(image.shape)
        # print(edge.shape)

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


if __name__ == '__main__':
    image_root = "/home/data/wangzeyu/FLIR_ADAS_1_3/train/thermal_8_bit/"
    edge_root = "/home/data/wangzeyu/FLIR_ADAS_1_3/train/edge/"
    mask_root = "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/testing_mask_dataset/"
    load_size = 288
    crop_size = 256
    dataset = TrainDataset(image_root, edge_root, mask_root, load_size, crop_size, return_image_root=False)
    masked_image, masked_edge, image, edge, mask = dataset[0]

    from torchvision.utils import save_image
    save_image(masked_image, './masked_image.jpg')
    save_image(masked_edge, './masked_edge.jpg')
    save_image(image, './image.jpg')
    save_image(edge, './edge.jpg')
    save_image(mask, './mask.jpg')
    print(len(dataset))
    print(masked_image.shape)