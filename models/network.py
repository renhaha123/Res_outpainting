import torch.nn as nn
import torch
import functools
from torch.nn import init
from util.util import assign_adain_params
import torch.nn.functional as F


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def define_G_S(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net_G = GeneratorS()

    return init_net(net_G, init_type, init_gain, gpu_ids)


def define_G(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net_G = Generator()

    return init_net(net_G, init_type, init_gain, gpu_ids)


def define_D(init_type='normal', init_gain=0.02, gpu_ids=[]):
    net_D = NLayerDiscriminator()

    return init_net(net_D, init_type, init_gain, gpu_ids)


class Block_E(nn.Module):
    def __init__(self, in_c, out_c, down=True):
        super(Block_E, self).__init__()

        self.down = down
        self.left = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
        )

        self.short_cut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c)
        )

        self.pool = nn.AvgPool2d(2, 2)

        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        if self.down:
            left = self.pool(self.left(x))
            shortcut = self.pool(self.short_cut(x))
        else:
            left = self.left(x)
            shortcut = self.short_cut(x)
        out = self.act(left + shortcut)

        return out


class Block_D(nn.Module):
    def __init__(self, in_c, out_c):
        super(Block_D, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(out_c, out_c, 3, 2, 1, 1),
            nn.InstanceNorm2d(out_c),
        )

        self.short_cut = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 3, 2, 1, 1),
            nn.InstanceNorm2d(out_c)
        )

        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):

        left = self.left(x)
        shortcut = self.short_cut(x)
        out = self.act(left + shortcut)

        return out


# Define Encoder
class EncoderS(nn.Module):
    def __init__(self, in_c=4, ngf=64):
        super(EncoderS, self).__init__()
        self.pre_block = nn.Sequential(
            nn.Conv2d(in_c, ngf, 3, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True)
        )
        self.encoder_0 = Block_E(ngf, ngf*2)
        self.encoder_1 = Block_E(ngf*2, ngf*4)
        self.encoder_2 = Block_E(ngf * 4, ngf * 8)
        self.encoder_3 = Block_E(ngf * 8, ngf * 8, False)
        self.encoder_4 = Block_E(ngf * 8, ngf * 8, False)

    def forward(self, x):
        out = self.pre_block(x)
        out = self.encoder_0(out)
        out = self.encoder_1(out)
        out = self.encoder_2(out)
        out = self.encoder_3(out)
        out = self.encoder_4(out)

        return out


# Define decoder
class DecoderS(nn.Module):
    def __init__(self, ngf=64):
        super(DecoderS, self).__init__()

        self.pre_block_0 = Block_E(ngf*8, ngf*8, False)
        self.pre_block_1 = Block_E(ngf * 8, ngf * 8, False)

        self.decoder_0 = Block_D(ngf*8, ngf*4)
        self.decoder_1 = Block_D(ngf * 4, ngf * 2)
        self.decoder_2 = Block_D(ngf * 2, ngf)
        self.out_block = nn.Sequential(
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, flag):
        out_ls = []
        out = self.pre_block_0(x)
        out_ls.append(out.detach())
        out = self.pre_block_1(out)
        out_ls.append(out.detach())
        out = self.decoder_0(out)
        out_ls.append(out.detach())
        out = self.decoder_1(out)
        out_ls.append(out.detach())
        out = self.decoder_2(out)
        out_ls.append(out.detach())
        out = self.out_block(out)
        out_ls.append(out.detach())

        if flag:
            return out_ls
        else:
            return out


# Define Encoder
class Encoder(nn.Module):
    def __init__(self, in_c=6, ngf=64):
        super(Encoder, self).__init__()
        self.pre_block = nn.Sequential(
            nn.Conv2d(in_c, ngf, 3, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True)
        )
        self.encoder_0 = Block_E(ngf, ngf*2)
        self.encoder_1 = Block_E(ngf*2, ngf*4)
        self.encoder_2 = Block_E(ngf * 4, ngf * 8)
        self.encoder_3 = Block_E(ngf * 8, ngf * 8, False)
        self.encoder_4 = Block_E(ngf * 8, ngf * 8, False)

        self.dia_blocks = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 2, 2),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 4, 4),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 8, 8),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_dia_1 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 2, 2),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_dia_2 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 4, 4),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_dia_3 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 8, 8),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(ngf * 8 * 3, ngf * 8, 1, 1),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        out_pre = self.pre_block(x)
        out_0 = self.encoder_0(out_pre)
        out_1 = self.encoder_1(out_0)
        out_2 = self.encoder_2(out_1)
        out_3 = self.encoder_3(out_2)
        out_4 = self.encoder_4(out_3)
        # out = self.dia_blocks(out)
        # print(out.shape)

        dia_1 = self.conv_dia_1(out_4)
        dia_2 = self.conv_dia_2(out_4)
        dia_3 = self.conv_dia_3(out_4)

        # print(dia_3.shape)

        out = self.fuse_conv(torch.cat([dia_1, dia_2, dia_3], dim=1))
        # print(out.shape)

        return out, out_4, out_2


# Define decoder
class Decoder(nn.Module):
    def __init__(self, ngf=64):
        super(Decoder, self).__init__()

        self.down_1_1 = Block_D(ngf * 4, ngf*2)
        self.down_1_2 = Block_D(ngf * 2, ngf)

        self.down_2_1 = Block_D(ngf * 2, ngf)

        self.down_3_1 = Block_E(ngf, ngf, False)

        self.fuse = nn.Sequential(
            nn.Conv2d(ngf * 3, ngf, 1, 1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2, True)
        )

        self.fuse_0 = Block_E(ngf, ngf*2)
        self.fuse_1 = Block_E(ngf*2, ngf*4)

        self.pre_block_0 = Block_E(ngf*8, ngf*8, False)
        self.pre_block_1 = Block_E(ngf * 8, ngf * 8, False)
        self.pre_block_2 = Block_E(ngf * 8, ngf * 8, False)
        self.decoder_0 = Block_D(ngf * 8, ngf * 4)
        self.decoder_1 = Block_D(ngf * 4 * 2, ngf * 2)
        self.decoder_2 = Block_D(ngf * 2 * 2, ngf)
        self.out_block = nn.Sequential(
            nn.Conv2d(ngf * 2, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, out_4, out_2, fea_s):

        f_1 = self.down_1_1(fea_s[-4])
        f_1 = self.down_1_2(f_1)

        f_2 = self.down_2_1(fea_s[-3])

        f_3 = self.down_3_1(fea_s[-2])

        f_fuse = self.fuse(torch.cat([f_1, f_2, f_3], dim=1))

        f_fuse_0 = self.fuse_0(f_fuse)

        f_fuse_1 = self.fuse_1(f_fuse_0)

        out = self.pre_block_0(x)
        out = self.pre_block_1(out)
        out = self.pre_block_2(out)
        out = self.decoder_0(out)
        out = self.decoder_1(torch.cat([out, f_fuse_1], dim=1))
        out = self.decoder_2(torch.cat([out, f_fuse_0], dim=1))
        out = self.out_block(torch.cat([out, f_fuse], dim=1))

        return out


# patch discriminator
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), False),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias), False),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), False),

            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias), True)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, fea_s):
        out_e, out_4, out_2 = self.encoder(x)
        out = self.decoder(out_e, out_4, out_2, fea_s)

        return out


class GeneratorS(nn.Module):
    def __init__(self):
        super(GeneratorS, self).__init__()

        self.encoder = EncoderS()
        self.decoder = DecoderS()

    def forward(self, x, flag=False):
        out_e = self.encoder(x)
        out = self.decoder(out_e, flag)

        return out

