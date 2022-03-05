import torch
import itertools
import random
import numpy as np
from .base_model import BaseModel
from . import networks3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # chin added 2022.01.28


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of '
                                     'scaling the weight of the identity mapping loss. For example, if the weight of the'
                                     ' identity loss should be 10 times smaller than the weight of the reconstruction loss, '
                                     'please set lambda_identity = 0.1')
            '''
            adjust the weight of correlation coefficient loss
            '''
            parser.add_argument('--lambda_co_A', type=float, default=2,
                                help='weight for correlation coefficient loss (A -> B)')
            parser.add_argument('--lambda_co_B', type=float, default=2,
                                help='weight for correlation coefficient loss (B -> A )')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'cor_coe_GA', 'D_B', 'G_B', 'cycle_B', 'cor_coe_GB']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        if not self.opt.coordconv:
            self.netG_A = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                              # nc number channels
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks3D.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netG_A = networks3D.define_G(4, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                              # nc number channels
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netG_B = networks3D.define_G(4, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if not self.opt.coordconv:
                self.netD_A = networks3D.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                  opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                                  self.gpu_ids)
                self.netD_B = networks3D.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                  opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                                  self.gpu_ids)
            else:
                self.netD_A = networks3D.define_D(1, opt.ndf, opt.netD,
                                                  opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                                  self.gpu_ids)
                self.netD_B = networks3D.define_D(1, opt.ndf, opt.netD,
                                                  opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                                  self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks3D.GANLoss(use_lsgan=not opt.no_lsgan, target_real_label=opt.real_label).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # Paired L1 loss
            self.pairedL1criterion = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.gen_lr, betas=(opt.gen_beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.disc_lr, betas=(opt.disc_beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Scalers
            self.gen_scaler = torch.cuda.amp.GradScaler()
            self.disc_scaler = torch.cuda.amp.GradScaler()

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        if self.opt.coordconv:
            self.coords = input[2].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A.to(device))
        self.fake_A = self.netG_B(self.real_B.to(device))

        if not self.opt.coordconv:
            self.rec_A = self.netG_B(self.fake_B.to(device))
            self.rec_B = self.netG_A(self.fake_A.to(device))
        else:
            self.rec_A = self.netG_B(torch.cat((self.fake_B.to(device), self.coords), dim=1))
            self.rec_B = self.netG_A(torch.cat((self.fake_A.to(device), self.coords), dim=1))

    def test_forward(self, overlap=0.3):
        from monai.inferers import sliding_window_inference
        fake_B = sliding_window_inference(self.real_A.to(device), 160, 1, self.netG_A,
                                          overlap=overlap,
                                          mode='gaussian')
        fake_A = sliding_window_inference(self.real_B.to(device), 160, 1, self.netG_B,
                                          overlap=overlap,
                                          mode='gaussian')
        if not self.opt.coordconv:
            rec_A = sliding_window_inference(fake_B.to(device), 160, 1, self.netG_B,
                                             overlap=overlap,
                                             mode='gaussian')
            rec_B = sliding_window_inference(fake_A.to(device), 160, 1, self.netG_A,
                                             overlap=overlap,
                                             mode='gaussian')
        else:
            rec_A = sliding_window_inference(torch.cat((fake_B.to(device), self.coords), dim=1), 160, 1, self.netG_B,
                                             overlap=overlap,
                                             mode='gaussian')
            rec_B = sliding_window_inference(torch.cat((fake_A.to(device), self.coords), dim=1), 160, 1, self.netG_A,
                                             overlap=overlap,
                                             mode='gaussian')
        return fake_B, rec_A, fake_A, rec_B

    def backward_D_basic(self, netD, real, fake, real_label_flip_chance=0.25):
        # print(f"The label flipping chance is {real_label_flip_chance}")
        # Real
        pred_real = netD(real[:, 0, ...][:, None, ...])
        flip_labels = np.random.uniform(0, 1)
        if flip_labels < real_label_flip_chance:
            loss_D_real = self.criterionGAN(pred_real, False)
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake[:, 0, ...][:, None, ...].detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D_A(self):
        # fake_B = self.fake_B_pool.query(self.fake_B)
        # self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        fake_B = self.fake_B_pool.query(self.fake_B.to(device))
        self.loss_D_A = self.backward_D_basic(self.netD_A.to(device), self.real_B.to(device), fake_B,
                                              real_label_flip_chance=self.opt.label_flipping_chance)

    def backward_D_B(self):
        # fake_A = self.fake_A_pool.query(self.fake_A)
        # self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        fake_A = self.fake_A_pool.query(self.fake_A.to(device))
        self.loss_D_B = self.backward_D_basic(self.netD_B.to(device), self.real_A.to(device), fake_A,
                                              real_label_flip_chance=self.opt.label_flipping_chance)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        '''
        lambda_coA & lambda_coB
        '''
        lambda_co_A = self.opt.lambda_co_A
        lambda_co_B = self.opt.lambda_co_B

        # Identity loss
        if lambda_idt > 0:
            ## old
            ## G_A should be identity if real_B is fed.
            # self.idt_A = self.netG_A(self.real_B)
            # self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            ## G_B should be identity if real_A is fed.
            # self.idt_B = self.netG_B(self.real_A)
            # self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            ## end of old

            # Chin
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B.to(device))
            self.loss_idt_A = self.criterionIdt(self.idt_A.to(device), self.real_B.to(device)) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A.to(device))
            self.loss_idt_B = self.criterionIdt(self.idt_B.to(device), self.real_A.to(device)) * lambda_A * lambda_idt
            # End of Chin
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) # chin commented 2022.01.28
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B.to(device)), True)  # chin added 2022.01.28

        # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) # chin commented 2022.01.28
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A.to(device)), True)  # chin added 2022.01.28

        # Forward cycle loss
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A # chin commented 2022.01.28
        self.loss_cycle_A = self.criterionCycle(self.rec_A.to(device),
                                                self.real_A.to(device)) * lambda_A  # chin added 2022.01.28

        # Backward cycle loss
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B # chin commented 2022.01.28
        self.loss_cycle_B = self.criterionCycle(self.rec_B.to(device),
                                                self.real_B.to(device)) * lambda_B  # chin added 2022.01.28

        '''
        self.cor_coeLoss
        '''
        # self.loss_cor_coe_GA = networks3D.Cor_CoeLoss(self.fake_B,self.real_A) * lambda_co_A  # fake ct & real mr; Evaluate the Generator of ct(G_A) # chin commented 2022.01.28
        # self.loss_cor_coe_GB = networks3D.Cor_CoeLoss(self.fake_A,self.real_B) * lambda_co_B  # fake mr & real ct; Evaluate the Generator of mr(G_B) # chin commented 2022.01.28
        self.loss_cor_coe_GA = networks3D.Cor_CoeLoss(self.fake_B.to(device), self.real_A.to(
            device)) * lambda_co_A  # fake ct & real mr; Evaluate the Generator of ct(G_A)  #chin added 2022.01.28
        self.loss_cor_coe_GB = networks3D.Cor_CoeLoss(self.fake_A.to(device), self.real_B.to(
            device)) * lambda_co_B  # fake mr & real ct; Evaluate the Generator of mr(G_B)  #chin added 2022.01.28

        # L1 loss
        if self.opt.paired_l1_loss:
            self.pairedL1Loss = self.pairedL1criterion(self.fake_B.to(device), self.real_B.to(device))

            # combined loss
            # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_cor_coe_GA + self.loss_cor_coe_GB + self.pairedL1Loss
        else:
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_cor_coe_GA + self.loss_cor_coe_GB

    def optimize_parameters(self, training=True):
        with torch.cuda.amp.autocast(enabled=True):
            # forward
            self.forward()
            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)
            self.optimizer_G.zero_grad()
            self.backward_G()
        if training:
            # Scale
            self.gen_scaler.scale(self.loss_G).backward()
            self.gen_scaler.step(self.optimizer_G)
            self.gen_scaler.update()
        # D_A and D_B
        with torch.cuda.amp.autocast(enabled=True):
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.backward_D_B()
        # Scale
        if training:
            self.disc_scaler.scale(self.loss_D_A).backward()
            self.disc_scaler.step(self.optimizer_D)
            self.disc_scaler.update()
            self.disc_scaler.scale(self.loss_D_B).backward()
            self.disc_scaler.step(self.optimizer_D)
            self.disc_scaler.update()