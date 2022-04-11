from options.base_options import BaseOptions
import argparse


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--csv_file', type=str, default="/nfs/home/pedro/MRA-GAN/IXI-MRA-GAN-fold-csv.csv", help='Dataset csv containing fold information')
        parser.add_argument('--real_label', type=float, default=1.0, help='Target real label for GAN loss: 1 by default but supports smoothing')
        parser.add_argument('--label_flipping_chance', type=float, default=0.25, help='Chance of label flipping')
        parser.add_argument('--coordconv', type=self.str2bool, nargs='?', const=True, default=False, help='Using coordconv or not')
        parser.add_argument('--job_name', type=str, default="", help='Job name')
        parser.add_argument('--augmentation_level', type=str, default="heavy", help='Use none, light, or heavy augmentation strategy')
        parser.add_argument('--paired_l1_loss', type=self.str2bool, nargs='?', const=True, default=False, help='Use a paired L1 loss between real and fake MRAs')
        parser.add_argument('--final_act', type=str, default="leaky", help='Activation for G_A network: i.e. MRA -> Vasc.')
        parser.add_argument('--perceptual', type=self.str2bool, nargs='?', const=True, default=False, help='Whether to use perceptual loss')
        parser.add_argument('--msssim', type=self.str2bool, nargs='?', const=True, default=False, help='Whether to use perceptual loss')
        parser.add_argument('--t1_aid', type=self.str2bool, nargs='?', const=True, default=False, help='Whether to use T1s as supplementary information')
        parser.add_argument('--znorm', type=self.str2bool, nargs='?', const=True, default=True, help='Whether to use znorm')
        parser.add_argument('--upsampling_method', type=str, default="trilinear", help='Activation for G_A network: i.e. MRA -> Vasc.')
        parser.add_argument('--lambda_cycle', type=int, default=1, help='Multiplicative factor for cycle losses')
        parser.add_argument('--perceptual_weighting', type=int, default=1, help='Multiplicative factor for perceptual loss')
        parser.add_argument('--msssim_weighting', type=int, default=1, help='Multiplicative factor for msssim loss')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
        # parser.add_argument('--patch_size', type=int, default=128, help='Patch size')
        #parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--gen_beta1', type=float, default=0.5, help='Generator momentum term of adam')
        parser.add_argument('--disc_beta1', type=float, default=0.5, help='Discriminator momentum term of adam')
        parser.add_argument('--gen_lr', type=float, default=0.0002, help='Generator initial learning rate for adam')
        parser.add_argument('--disc_lr', type=float, default=0.0002, help='Discriminator initial learning rate for adam')
        parser.add_argument('--no_lsgan', type=self.str2bool, nargs='?', const=True, default=False, help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser

    # Function for proper handling of bools in argparse
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')



