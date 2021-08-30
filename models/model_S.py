import torch.nn as nn
import torch
from models.network import define_G_S, define_D, print_network
from util.util import unfreeze_network, freeze_network, get_scheduler, cal_gradient_penalty
from models.loss import GANLoss
import os


class Model_S(nn.Module):
    def __init__(self, opt):
        super(Model_S, self).__init__()

        self.opt = opt
        self.save_dir = self.opt.save_dir_rtv

        self.net_G = define_G_S(init_type=self.opt.init_type, gpu_ids=self.opt.gpu_ids)
        self.net_D = define_D(init_type=self.opt.init_type, gpu_ids=self.opt.gpu_ids)

        self.model_names = ['G', 'D']

        if opt.train_rtv:
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            for optimizer in self.optimizers:
                self.schedulers.append(get_scheduler(optimizer, opt))

            print('Initialize network:')
            print_network(self.net_G)
            print_network(self.net_D)

            self.Ganloss = GANLoss(opt.gan_mode)
            self.L1_Loss = torch.nn.L1Loss()
            self.L2_Loss = torch.nn.MSELoss()
            # self.Perc_Loss = PerceptualLoss().to('cuda')
            # self.Style_Loss = StyleLoss().to('cuda')

        if self.opt.ckpt_iter_rtv:
            self.load_network()

    def set_input(self, imgs, masks):
        self.img_gt = imgs.to('cuda')
        self.mask = masks.to('cuda')

        self.masked_img = self.mask * self.img_gt

    def forward(self):
        self.img_gen = self.net_G(torch.cat([self.masked_img, 1-self.mask], 1))

    def backward_D(self):
        unfreeze_network(self.net_D)
        fake_score = self.net_D(self.img_gen.detach())
        fake_loss_D = self.Ganloss(fake_score, False, True)

        real_score = self.net_D(self.img_gt)
        real_loss_D = self.Ganloss(real_score, True, True)

        self.loss_D = (fake_loss_D + real_loss_D) * 0.5

        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = cal_gradient_penalty(self.net_D, self.img_gt, self.img_gen.detach())
            self.loss_D += gradient_penalty

        self.loss_D.backward()

    def backward_G(self):
        freeze_network(self.net_D)
        fake_score = self.net_D(self.img_gen)
        self.fake_loss_G = self.Ganloss(fake_score, True, False) * self.opt.lambda_G

        self.rec_loss = self.L1_Loss(self.img_gen, self.img_gt) * self.opt.lambda_rec

        # self.perc_loss = self.Perc_Loss(self.img_gen, self.img_gt) * self.opt.lambda_perc
        # self.style_loss = self.Style_Loss(self.img_gen, self.img_gt) * self.opt.lambda_style

        self.loss_G = self.fake_loss_G + self.rec_loss

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.net_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.net_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_fea_s(self):
        out_ls = self.net_G(torch.cat([self.masked_img, 1-self.mask], 1), True)

        return out_ls, (self.img_gt + 1.0) / 2.0, (self.masked_img + 1.0) / 2.0

    def test(self):
        img_out = self.net_G(torch.cat([self.masked_img, 1 - self.mask], 1))
        return img_out

    def get_loss(self):
        return [self.loss_D, self.fake_loss_G, self.rec_loss]

    def get_result(self):
        img_gt = (self.img_gt + 1.0) / 2.0
        masked_img = (self.masked_img + 1.0) / 2.0
        img_gen = (self.img_gen + 1.0) / 2.0

        return img_gt.data, masked_img.data, img_gen.data

    def save_network(self, iter):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (iter, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)
                torch.save(net.cpu().state_dict(), save_path)
                if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

    def load_network(self):
        for name in self.model_names:
            file_name = '%s_net_%s.pth' % (self.opt.ckpt_iter_rtv, name)
            load_path = os.path.join(self.save_dir, file_name)
            net = getattr(self, 'net_' + name)
            try:
                net.load_state_dict(torch.load(load_path))
                print('load net_%s_s successfully' % name)
            except:
                print('load net_%s error' % name)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
