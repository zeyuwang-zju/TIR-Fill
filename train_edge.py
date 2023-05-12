import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from dataset import TrainDataset
from models.edge_connect import EdgeGenerator
from models.discriminator import Discriminator
from loss import AdversarialLoss, StyleLoss
from options.train_options import TrainOptions
from utils import seed_torch

def initialize_model(opt, device):
    edge_generator = EdgeGenerator(residual_blocks=8, use_spectral_norm=True, init_weights=True).to(device)
    discriminator = Discriminator(in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True).to(device)
    if opt.edge_ckpt_path is not None and os.path.exists(opt.edge_ckpt_path):
        edge_generator.load_state_dict(torch.load(opt.edge_ckpt_path)["edge_generator"])
        discriminator.load_state_dict(torch.load(opt.edge_ckpt_path)["edge_discriminator"])
        start_epoch = torch.load(opt.edge_ckpt_path)["edge_epoch"]
    else:
        start_epoch = 0
    
    return edge_generator, discriminator, start_epoch


if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(opt.seed)

    dataset_name = opt.dataset_name
    sample_dir = f'samples/{dataset_name}/edge'
    ckpt_dir = f'checkpoints/{dataset_name}/edge'
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    sample_step = opt.sample_step
    sample_size = opt.sample_size

    image_root = opt.image_root # "/home/data/wangzeyu/FLIR_ADAS_1_3/train/thermal_8_bit/"
    edge_root = opt.edge_root   # "/home/data/wangzeyu/FLIR_ADAS_1_3/train/edge/"
    mask_root = opt.mask_root   # "/home/data/wangzeyu/Image_Inpainting/mask_pconv/test_mask/testing_mask_dataset/"
    load_size = opt.loadsize
    crop_size = opt.cropsize
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    dataset = TrainDataset(image_root, edge_root, mask_root, load_size, crop_size, return_image_root=False)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
    device = torch.device(opt.device)
    lr = opt.lr
    beta1 = opt.beta1
    beta2 = opt.beta2
    num_epochs = opt.num_epochs
    num_batch = len(dataloader)

    l1 = nn.L1Loss()
    adversarial_loss = AdversarialLoss(type='hinge')
    fm_loss_weight = opt.fm_loss_weight

    # edge_generator = EdgeGenerator(residual_blocks=8, use_spectral_norm=True, init_weights=True).to(device)
    # discriminator = Discriminator(in_channels=2, use_sigmoid=False, use_spectral_norm=True, init_weights=True).to(device)

    edge_generator, discriminator, start_epoch = initialize_model(opt, device)

    optimizer_G = optim.Adam(edge_generator.parameters(), lr, betas=[beta1, beta2])
    optimizer_D = optim.Adam(discriminator.parameters(), lr, betas=[beta1, beta2])

    for epoch in range(start_epoch, num_epochs):
        for batch_idx, (masked_image, masked_edge, image, edge, mask) in enumerate(dataloader):
            # print(masked_image.shape)
            # exit()
            masked_image, masked_edge, image, edge, mask = masked_image.to(device), masked_edge.to(device), image.to(device), edge.to(device), mask.to(device)
            outputs = edge_generator(masked_image, masked_edge, mask)
            gen_loss = 0
            dis_loss = 0

            # discriminator loss
            dis_input_real = torch.cat((image, mask, edge), dim=1)
            dis_input_fake = torch.cat((image, mask, outputs.detach()), dim=1)
            dis_real, dis_real_feat = discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
            dis_fake, dis_fake_feat = discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
            dis_real_loss = adversarial_loss(dis_real, True, True)
            dis_fake_loss = adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

            # dis_real_loss = torch.mean(F.relu(1. - dis_real))
            # dis_fake_loss = torch.mean(F.relu(1. + dis_fake))
            # dis_loss += 0.5 * (dis_real_loss + dis_fake_loss)


            # generator adversarial loss
            gen_input_fake = torch.cat((image, mask, outputs), dim=1)
            gen_fake, gen_fake_feat = discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
            gen_gan_loss = adversarial_loss(gen_fake, True, False)
            gen_loss += gen_gan_loss


            # # generator feature matching loss
            # gen_fm_loss = 0
            # for i in range(len(dis_real_feat)):
            #     gen_fm_loss += l1(gen_fake_feat[i], dis_real_feat[i].detach())
            # gen_fm_loss = gen_fm_loss * fm_loss_weight
            # gen_loss += gen_fm_loss

            optimizer_G.zero_grad()
            gen_loss.backward(retain_graph=True)
            optimizer_D.zero_grad()
            dis_loss.backward()
            optimizer_G.step()
            optimizer_D.step()

            print(f'Epoch[{epoch}/{num_epochs}] | Batch[{batch_idx}/{num_batch}] | G_adv_loss: {gen_gan_loss:.3f} | D_loss: {dis_loss:.3f} | dis_real: {dis_real.mean():.3f} | dis_fake: {dis_fake.mean():.3f}')

            if batch_idx % sample_step == 0:
                threshold = 0.5
                ones = outputs >= threshold
                zeros = outputs < threshold
                outputs.masked_fill_(ones, 1.0)
                outputs.masked_fill_(zeros, 0.0)

                recom = mask * edge + (1 - mask) * outputs

                save_list = torch.cat((masked_edge[:sample_size], outputs[:sample_size], recom[:sample_size], edge[:sample_size], image[:sample_size]))
                save_image(save_list, os.path.join(sample_dir, f'epoch_{epoch}.jpg'), nrow=4)

        if (epoch + 1) % 50 == 0:
            ckpt = {'edge_epoch': epoch+1, 'edge_generator': edge_generator.state_dict(), 'edge_discriminator': discriminator.state_dict()}
            torch.save(ckpt, os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth'))


