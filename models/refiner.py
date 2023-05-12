import os, sys
sys.path.append(os.getcwd())
import torch.nn as nn
import torch

from models.base_model import BaseNetwork, spectral_norm, get_norm_layer, get_act_layer, Downsample, Upsample

#-----------------------------------------------
#                Gated ConvBlock
#-----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', norm_type='instance', act_type='leaky', use_spectral_norm=True):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        self.norm = get_norm_layer(norm_type)(out_channels)
        self.activation = get_act_layer(act_type)

        self.conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation), use_spectral_norm)
        self.mask_conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation), use_spectral_norm)
        self.sigmoid = nn.Sigmoid()
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
    
    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        x = self.pad(inputs)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        x = self.norm(x)
        x = self.activation(x) + shortcut
        return x

class Refined_Generator(BaseNetwork):
    def __init__(self, residual_blocks=8, norm_type='instance', act_type='leaky', use_spectral_norm=True, init_weights=True):
        super(Refined_Generator, self).__init__()
        self.residual_blocks = residual_blocks
        self.conv_in = spectral_norm(nn.Conv2d(2, 64, 7, 1, 3), mode=use_spectral_norm)

        self.enc_1 = GatedConv2d(64, 128, 3, 1, 1, norm_type=norm_type, act_type=act_type, use_spectral_norm=use_spectral_norm)
        self.enc_down1 = Downsample(128)
        self.enc_2 = GatedConv2d(128, 256, 3, 1, 1, norm_type=norm_type, act_type=act_type, use_spectral_norm=use_spectral_norm)
        self.enc_down2 = Downsample(256)
        self.enc_3 = GatedConv2d(256, 512, 3, 1, 1, norm_type=norm_type, act_type=act_type, use_spectral_norm=use_spectral_norm)
        self.enc_down3 = Downsample(512)
        self.enc_4 = GatedConv2d(512, 1024, 3, 1, 1, norm_type=norm_type, act_type=act_type, use_spectral_norm=use_spectral_norm)
        self.enc_down4 = Downsample(1024)

        for i in range(residual_blocks):
            setattr(self, f'mid_{i+1}', GatedConv2d(1024, 1024, 3, 1, 1, norm_type=norm_type, act_type=act_type, use_spectral_norm=use_spectral_norm))
        
        self.dec_up4 = Upsample(1024)
        self.dec_4 = GatedConv2d(1024, 512, 3, 1, 1, norm_type=norm_type, act_type=act_type, use_spectral_norm=use_spectral_norm)
        self.dec_up3 = Upsample(512)
        self.dec_3 = GatedConv2d(512, 256, 3, 1, 1, norm_type=norm_type, act_type=act_type, use_spectral_norm=use_spectral_norm)
        self.dec_up2 = Upsample(256)
        self.dec_2 = GatedConv2d(256, 128, 3, 1, 1, norm_type=norm_type, act_type=act_type, use_spectral_norm=use_spectral_norm)
        self.dec_up1 = Upsample(128)
        self.dec_1 = GatedConv2d(128, 64, 3, 1, 1, norm_type=norm_type, act_type=act_type, use_spectral_norm=use_spectral_norm)
        self.conv_out = spectral_norm(nn.Conv2d(64, 1, 7, 1, 3), mode=use_spectral_norm)
        
        if init_weights:
            self.init_weights()
        
    def forward(self, image, mask):
        out = self.conv_in(torch.cat((image, mask), dim=1))
        out = self.enc_1(out)
        out = self.enc_down1(out)
        out = self.enc_2(out)
        out = self.enc_down2(out)
        out = self.enc_3(out)
        out = self.enc_down3(out)
        out = self.enc_4(out)
        out = self.enc_down4(out)
        for i in range(self.residual_blocks):
            out = getattr(self, f'mid_{i+1}')(out)
        out = self.dec_up4(out) 
        out = self.dec_4(out)
        out = self.dec_up3(out)
        out = self.dec_3(out)
        out = self.dec_up2(out)
        out = self.dec_2(out)
        out = self.dec_up1(out)
        out = self.dec_1(out)
        out = self.conv_out(out)

        out = torch.sigmoid(out)

        return out

if __name__ == '__main__':
    refiner = Refiner(residual_blocks=8, norm_type='instance', act_type='leaky', use_spectral_norm=True, init_weights=True)
    print(refiner)
    image = torch.randn(size=(2, 1, 256, 256))
    # edge = torch.randn(size=(2, 1, 256, 256))
    mask = torch.randn(size=(2, 1, 256, 256))
    outputs = refiner(image, mask)
    print(outputs.shape)


        

