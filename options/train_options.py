from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True

        # Initialization settings
        self.parser.add_argument('--seed', type=int, default=0, help='Random Seed before Training')
        self.parser.add_argument('--device', type=str, help='Device [cuda:0, cuda:1..., cpu]')

        # # Dataset settings
        self.parser.add_argument('--image_root', type=str, help='The root of training images')
        self.parser.add_argument('--edge_root', type=str, help='The root of edges of training images')
        self.parser.add_argument('--mask_root', type=str, help='The root of masks')
        self.parser.add_argument('--loadsize', type=int, default=288, help='load images this size')
        self.parser.add_argument('--cropsize', type=int, default=256, help='then crop to this size [H, W]')
        
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
        self.parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
        self.parser.add_argument('--num_epochs', type=int, default=1000, help='number of training epochs')

        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for the Adam optimizer')
        self.parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for the Adam optimizer')

        self.parser.add_argument('--fm_loss_weight', type=float, default=1., help='factor of the feature matching loss')

        self.parser.add_argument('--sample_step', type=int, default=100, help='how many steps to sample once during training')
        self.parser.add_argument('--sample_size', type=int, default=4, help='how many images to sample during training')

        self.parser.add_argument('--edge_ckpt_path', type=str, help='The checkpoint path of pretrained edge-connect models')
        self.parser.add_argument('--gen_ckpt_path', type=str, help='The checkpoint path of pretrained generator models')
