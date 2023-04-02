import torch
import os
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
import torchvision.transforms as T
from . import networks
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B','D_Ac', 'G_Ac', 'D_Bc', 'G_Bc']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', "real_gt_A", "fake_gt_A"]
        visual_names_B = ['real_B', 'fake_A', "real_gt_B", "fake_gt_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
    
        if self.isTrain:  # define discriminators

            # Whole image discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            # Partial image discriminators
            self.netD_Ac = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_Bc = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        self.netC_A = networks.define_UNet(opt.A_domain_segmentor, opt.input_nc)
        self.netC_B = networks.define_UNet(opt.B_domain_segmentor, opt.output_nc)
         
        print("FUCKING OK")
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionSeg = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(),self.netD_Ac.parameters(), self.netD_Bc.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_gt_A_cell = input['A_gt_cell' if AtoB else 'B_gt_cell'].to(self.device)
        self.real_gt_B_cell = input['B_gt_cell' if AtoB else 'A_gt_cell'].to(self.device)
        self.real_gt_A_line = input['A_gt_line' if AtoB else 'B_gt_line'].to(self.device)
        self.real_gt_B_line = input['B_gt_line' if AtoB else 'A_gt_line'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        batchsize, A_ch, h, w = self.real_A.size()
        _, B_ch, _, _ = self.real_B.size()

        # combline cell and line ground truth in A domain
        self.real_gt_A = self.real_gt_A_line.clone()
        self.real_gt_A[self.real_gt_A_cell==1]=1
        # combline cell and line ground truth in B domain
        self.real_gt_B = self.real_gt_B_line.clone()
        self.real_gt_B[self.real_gt_B_cell==1]=1
        
        # real A --> fake B
        noise = torch.rand(batchsize, A_ch, h, w).to(self.device)
        self.noise_real_A = self.real_A + (noise-0.5)/9
        self.noise_real_A[self.noise_real_A > 1] = 1
        self.noise_real_A[self.noise_real_A < -1] = -1
        self.fake_B, self.fake_B_seg = self.netG_A(self.noise_real_A)  # G_A(A)
        # real B --> fake A
        noise = torch.rand(batchsize, B_ch, h, w).to(self.device)
        self.noise_real_B = self.real_B + (noise-0.5)/9
        self.noise_real_B[self.noise_real_B > 1] = 1
        self.noise_real_B[self.noise_real_B < -1] = -1
        self.fake_A, self.fake_A_seg = self.netG_B(self.noise_real_B)  # G_B(B)
        
        # predict the cell nuclei of fake A image
        cell_pred_A_temp, _ , _ = self.netC_A((self.fake_A+1)/2)
        self.cell_pred_A = 2*(torch.unsqueeze(torch.argmax(cell_pred_A_temp, dim=1),1).to(dtype = torch.float))-1

        # predict the cell nuclei of fake B image
        cell_pred_B_temp, _ , _ = self.netC_B((self.fake_B+1)/2)
        self.cell_pred_B = 2*(torch.unsqueeze(torch.argmax(cell_pred_B_temp, dim=1),1).to(dtype = torch.float))-1
        
        # combline cell prediction and line ground truth in A domain
        self.fake_gt_A = self.real_gt_B_line.clone()
        self.fake_gt_A[self.cell_pred_A==1]=1
        # combline cell prediction and line ground truth in B domain
        self.fake_gt_B = self.real_gt_A_line.clone()
        self.fake_gt_B[self.cell_pred_B==1]=1
        
        
        # fake B --> rec A
        noise = torch.rand(batchsize, B_ch, h, w).to(self.device)
        self.noise_fake_B = self.fake_B + (noise-0.5)/9
        self.noise_fake_B[self.noise_fake_B > 1] = 1
        self.noise_fake_B[self.noise_fake_B < -1] = -1
        self.rec_A, self.rec_A_seg = self.netG_B(self.noise_fake_B)   # G_B(G_A(A))
        # fake A --> rec B
        noise = torch.rand(batchsize, A_ch, h, w).to(self.device)
        self.noise_fake_A = self.fake_A + (noise-0.5)/9
        self.noise_fake_A[self.noise_fake_A > 1] = 1
        self.noise_fake_A[self.noise_fake_A < -1] = -1
        self.rec_B, self.rec_B_seg = self.netG_A(self.noise_fake_A)   # G_A(G_B(B))
        
        B_MIN = self.real_gt_B_line.min()
        B_MAX = self.real_gt_B_line.max()
        A_MIN = self.real_gt_A_line.min()
        A_MAX = self.real_gt_A_line.max()
        
        if(B_MAX<0):
            B_MAX = 2
        if(A_MAX<0):
            A_MAX = 2
        
        # partial real B
        self.real_B_cell = self.real_B.clone()
        self.real_B_cell[(self.real_gt_B_cell==1) | (self.real_gt_B_line==B_MIN) | (self.real_gt_B_line==B_MAX)] = -1
        # partial fake B
        self.fake_B_cell = self.fake_B.clone()
        self.fake_B_cell[(self.cell_pred_B==1) | (self.real_gt_A_line==A_MIN) | (self.real_gt_A_line==A_MAX)] = -1
  
        # partial real A
        self.real_A_cell = self.real_A.clone()
        self.real_gt_A_line = torch.cat((self.real_gt_A_line,self.real_gt_A_line,self.real_gt_A_line), dim=1)
        self.real_gt_A_cell = torch.cat((self.real_gt_A_cell,self.real_gt_A_cell,self.real_gt_A_cell), dim=1)
        self.real_A_cell[(self.real_gt_A_cell==1) | (self.real_gt_A_line==A_MIN) | (self.real_gt_A_line==A_MAX)] = -1
        
        # partial fake A
        self.fake_A_cell = self.fake_A.clone()
        self.real_gt_B_line = torch.cat((self.real_gt_B_line,self.real_gt_B_line,self.real_gt_B_line), dim=1)
        self.cell_pred_A = torch.cat((self.cell_pred_A,self.cell_pred_A,self.cell_pred_A), dim=1)
        self.fake_A_cell[(self.cell_pred_A==1) | (self.real_gt_B_line==B_MIN) | (self.real_gt_B_line==B_MAX)] = -1

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) 
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) 
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
        
    def backward_D_Ac(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_Ac = self.backward_D_basic(self.netD_Ac, self.real_B_cell, self.fake_B_cell)

    def backward_D_Bc(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_Bc = self.backward_D_basic(self.netD_Bc, self.real_A_cell, self.fake_A_cell)
    

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle-consistency loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # Supervised loss
        self.loss_seg_A = self.criterionSeg(self.fake_A_seg, self.fake_gt_A) * lambda_A
        self.loss_seg_B = self.criterionSeg(self.fake_B_seg, self.fake_gt_B) * lambda_B

        # Cycle-consistency label loss
        self.loss_rec_A = self.criterionSeg(self.rec_A_seg, self.real_gt_A) * lambda_A
        self.loss_rec_B = self.criterionSeg(self.rec_B_seg, self.real_gt_B) * lambda_B
        
        # Partial GAN loss
        self.loss_G_Ac = self.criterionGAN(self.netD_Ac(self.fake_B_cell), True) 
        self.loss_G_Bc = self.criterionGAN(self.netD_Bc(self.fake_A_cell), True)

        # All together
        self.loss_G1 = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G2 = self.loss_seg_A + self.loss_seg_B + self.loss_rec_A + self.loss_rec_B 
        self.loss_G3 = self.loss_G_Ac + self.loss_G_Bc 
        
        self.loss_G = self.loss_G1 + self.loss_G2 + self.loss_G3
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_Ac, self.netD_Bc], False)  # Ds require no gradients when optimizing Gs
        #self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_Ac, self.netD_Bc], True)
        #self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.backward_D_Ac()      # calculate gradients for D_Ac
        self.backward_D_Bc()      # calculate graidents for D_Bc
        self.optimizer_D.step()  # update D_A and D_B's weights


