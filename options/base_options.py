import argparse
import os
# from util import util
# import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataset_name', required=True, type=str, help='name of the experiment. It decides where to store samples and models')
        # self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # self.parser.add_argument('--dataroot', required=True, type=str, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        # self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize|resize_and_crop|crop]')

        # self.parser.add_argument('--NTIR_channel', type=int, default=1, help='The number of NTIR image channels')
        # self.parser.add_argument('--DC_channel', type=int, default=3, help='The number of DC image channels')
        # self.parser.add_argument('--edge_channel', type=int, default=1, help='The number of edge channels')

        # # model settings
        # self.parser.add_argument('--norm_type', type=str, default='instance', help='Normlization Type [batch | instance | group | none]')
        # self.parser.add_argument('--act_type', type=str, default='leaky', help='Activation Type [relu | leaky | swish | none]')
        # self.parser.add_argument('--spec_norm', type=int, default=1, help='whether to use spectral norm on Conv2d [0: not| 1: yes]')


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join('./checkpoints', self.opt.dataset_name)
        # util.mkdirs(expr_dir)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt