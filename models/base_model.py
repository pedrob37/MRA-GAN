import os
import torch
from collections import OrderedDict
from models import networks3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # chin added 20220128


class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:0') if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        if self.isTrain:
            self.schedulers = [networks3D.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # make models eval mode during test time
    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # Save models to the disk
    def save_networks(self, which_epoch, current_iter, models_dir):
        # Define ONE file for saving ALL state dicts
        save_filename = f'epoch_{which_epoch}_checkpoint_iters_{current_iter}.pth'
        current_state_dict = {'gen_optimizer_state_dict': self.optimizer_G.state_dict(),
                              'disc_optimizer_state_dict': self.optimizer_D.state_dict(),
                              'epoch': which_epoch,
                              'running_iter': current_iter,
                              'batch_size': self.opt.batch_size,
                              'patch_size': self.opt.patch_size,
                              'gen_scaler': self.gen_scaler.state_dict(),
                              'disc_scaler': self.disc_scaler.state_dict()}
        for name in self.model_names:
            if isinstance(name, str):
                save_path = os.path.join(models_dir, save_filename)
                net = getattr(self, 'net' + name)
                net.cpu()
                if torch.cuda.is_available():
                    current_state_dict[f'net_{name}_state_dict'] = net.state_dict()
                    net.cuda(device)
        # Save aggregated checkpoint file
        torch.save(current_state_dict, save_path)

    def write_logs(self, training=True, step=None, current_writer=None):
        losses = self.get_current_losses()
        if training:
            current_writer.add_scalars('Loss/Adversarial',
                                       {"Generator_A": losses["G_A"],
                                        "Generator_B": losses["G_B"],
                                        "Discriminator_A": losses["D_A"],
                                        "Discriminator_B": losses["D_B"]}, step)
        else:
            current_writer.add_scalars('Loss/Val_Adversarial',
                                       {"Generator_A": losses["G_A"],
                                        "Generator_B": losses["G_B"],
                                        "Discriminator_A": losses["D_A"],
                                        "Discriminator_B": losses["D_B"]}, step)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        print('keys', keys)
        print('key', key)
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys,
                                                  i + 1)  # Chin commented 2022.02.02

    # load models from the disk
    def load_networks(self, which_epoch, models_dir, phase="train"):
        import glob
        if which_epoch == "latest":
            # Find latest epoch
            model_files = glob.glob(os.path.join(models_dir, '*.pth'))
            print("The following model files were found:\n")
            for some_model_file in model_files:
                print(some_model_file)
            sorted_model_files = sorted(model_files, key=os.path.getmtime)
            # Allows inference to be run on nth latest file!
            load_path = sorted_model_files[-1]
        else:
            # Loading ONE torch file for all nets
            load_filename = f'epoch_{which_epoch}_checkpoint*'
            load_path = glob.glob(os.path.join(models_dir, load_filename))[0]
        print(f'Loading the model from {load_path}')
        checkpoint = torch.load(load_path, map_location=self.device)
        if phase == "train":
            # Scalers and optimizers
            self.optimizer_G.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            self.gen_scaler.load_state_dict(checkpoint['gen_scaler'])
            self.disc_scaler.load_state_dict(checkpoint['disc_scaler'])

        # Networks loading
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                if hasattr(checkpoint[f"net_{name}_state_dict"], '_metadata'):
                    del checkpoint[f"net_{name}_state_dict"]._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(checkpoint[
                                    f"net_{name}_state_dict"].keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(checkpoint["model_state_dict"], net, key.split('.'))
                net.load_state_dict(checkpoint[f"net_{name}_state_dict"])

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
