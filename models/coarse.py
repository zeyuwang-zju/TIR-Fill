import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseNetwork, spectral_norm, get_norm_layer, get_act_layer, Downsample, Upsample


class SPADE(nn.Module):
    def __init__(self, feature_channels, norm_type='instance', act_type='leaky', use_spectral_norm=True):
        super(SPADE, self).__init__()
        self.norm_feature = get_norm_layer(norm_type)(feature_channels)
        self.conv_edge = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 128, 3, 1, 1), mode=use_spectral_norm),
            get_norm_layer(norm_type)(128),
            get_act_layer(act_type)
        )
        self.conv_beta = spectral_norm(nn.Conv2d(128, feature_channels, 3, 1, 1), mode=use_spectral_norm)
        self.conv_gamma = spectral_norm(nn.Conv2d(128, feature_channels, 3, 1, 1), mode=use_spectral_norm)

    def forward(self, feature, edge):
        feature = self.norm_feature(feature)

        edge = F.interpolate(edge, size=(feature.shape[2], feature.shape[3]), mode='nearest')
        edge = self.conv_edge(edge)

        beta = self.conv_beta(edge)
        gamma = self.conv_gamma(edge)

        feature = feature * gamma + beta

        return feature


class SPADE_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='instance', act_type='leaky', use_spectral_norm=True):
        super(SPADE_Resblock, self).__init__()
        self.spade1 = SPADE(in_channels, norm_type, act_type, use_spectral_norm)
        self.act1 = get_act_layer(act_type)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1), mode=use_spectral_norm)
        self.spade2 = SPADE(out_channels, norm_type, act_type, use_spectral_norm)
        self.act2 = get_act_layer(act_type)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1), mode=use_spectral_norm)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, feature, edge):
        shortcut = self.shortcut(feature)
        out = self.spade1(feature, edge)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.spade2(out, edge)
        out = self.act2(out)
        out = self.conv2(out)

        # if self.shortcut == None:
        #     shortcut = feature
        # else:
        #     shortcut = self.shortcut(feature)
        
        return out + shortcut


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='instance', act_type='leaky', use_spectral_norm=True):
        super(Block, self).__init__()
        self.conv_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            get_norm_layer(norm_type)(out_channels),
            get_act_layer(act_type),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            get_norm_layer(norm_type)(out_channels),
            get_act_layer(act_type),
        )

        # if in_channels == out_channels:
        #     self.shortcut = None
        # else:
        #     self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = self.conv_block(x)

        # if self.shortcut == None:
        #     shortcut = x
        # else:
        #     shortcut = self.shortcut(x)
        # out = out + shortcut

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


# class Downsample(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv2d(channels, channels, 3, 2, 1)

#     def forward(self, x):
#         x = self.conv(x)

#         return x

    
# class Upsample(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=2)
#         x = self.conv(x)

#         return x



class Coarse_Generator(BaseNetwork):
    def __init__(self, residual_blocks=8, norm_type='instance', act_type='leaky', use_spectral_norm=True, init_weights=True):
        super(Coarse_Generator, self).__init__()
        self.residual_blocks = residual_blocks

        self.conv_in = spectral_norm(nn.Conv2d(2, 64, 7, 1, 3), mode=use_spectral_norm)
        self.enc_res1 = Block(64, 128, norm_type, act_type, use_spectral_norm)
        self.enc_down1 = Downsample(128)
        self.enc_res2 = Block(128, 256, norm_type, act_type, use_spectral_norm)
        self.enc_down2 = Downsample(256)
        self.enc_res3 = Block(256, 512, norm_type, act_type, use_spectral_norm)
        self.enc_down3 = Downsample(512)
        self.enc_res4 = Block(512, 1024, norm_type, act_type, use_spectral_norm)
        self.enc_down4 = Downsample(1024)

        for i in range(residual_blocks):
            setattr(self, f'mid_spade{i+1}', SPADE_Resblock(1024, 1024, norm_type, act_type, use_spectral_norm))

        self.dec_up4 = Upsample(1024)
        self.dec_res4 = Block(2048, 512, norm_type, act_type, use_spectral_norm)
        self.dec_up3 = Upsample(512)
        self.dec_res3 = Block(1024, 256, norm_type, act_type, use_spectral_norm)
        self.dec_up2 = Upsample(256)
        self.dec_res2 = Block(512, 128, norm_type, act_type, use_spectral_norm)
        self.dec_up1 = Upsample(128)
        self.dec_res1 = Block(256, 64, norm_type, act_type, use_spectral_norm)
        self.conv_out = spectral_norm(nn.Conv2d(64, 1, 7, 1, 3), mode=use_spectral_norm)

        if init_weights:
            self.init_weights()

    def forward(self, masked_image, edge, mask):

        e1 = self.conv_in(torch.cat((masked_image, mask), dim=1))   # (B, 64, H, W)
        e1 = self.enc_res1(e1)                                         # (B, 128, H, W)
        e2 = self.enc_down1(e1)                                     # (B, 128, H//2, W//2)
        e2 = self.enc_res2(e2)                                         # (B, 256, H//2, W//2)
        e3 = self.enc_down2(e2)                                     # (B, 256, H//4, W//4)
        e3 = self.enc_res3(e3)                                         # (B, 512, H//4, W//4)
        e4 = self.enc_down3(e3)                                     # (B, 512, H//8, W//8)
        e4 = self.enc_res4(e4)                                         # (B, 1024, H//8, W//8)
        mid = self.enc_down4(e4)                                    # (B, 1024, H//16, W//16)

        for i in range(self.residual_blocks):
            mid = getattr(self, f'mid_spade{i+1}')(mid, edge)       # (B, 1024, H//16, W//16)
        
        d4 = self.dec_up4(mid)                                      # (B, 1024, H//8, W//8)
        d4 = self.dec_res4(torch.cat((d4, e4), dim=1))                 # (B, 512, H//8, W//8)
        d3 = self.dec_up3(d4)                                       # (B, 512, H//4, W//4)
        d3 = self.dec_res3(torch.cat((d3, e3), dim=1))                 # (B, 256, H//4, W//4)
        d2 = self.dec_up2(d3)                                       # (B, 256, H//2, W//2)
        d2 = self.dec_res2(torch.cat((d2, e2), dim=1))                 # (B, 128, H//2, W//2)
        d1 = self.dec_up1(d2)                                       # (B, 128, H, W)
        d1 = self.dec_res1(torch.cat((d1, e1), dim=1))                 # (B, 64, H, W)

        out = self.conv_out(d1)                                     # (B, 1, H, W)
        out = torch.sigmoid(out)

        return out


if __name__ == '__main__':
    generator = Generator(residual_blocks=8, norm_type='instance', act_type='leaky', use_spectral_norm=True, init_weights=True)
    print(generator)
    masked_image = torch.randn(size=(2, 1, 256, 256))
    edge = torch.randn(size=(2, 1, 256, 256))
    mask = torch.randn(size=(2, 1, 256, 256))
    outputs = generator(masked_image, edge, mask)
    print(outputs.shape)