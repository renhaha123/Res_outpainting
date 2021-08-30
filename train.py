import torch
import argparse
from data.dataset import Places2
from tensorboardX import SummaryWriter
import torchvision
from models.model import OutpaintingModel
from models.model_S import Model_S
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', type=str, default='/home/renyongpeng/dataset/places128/train', help='the path of img')
    parser.add_argument('--rtv_img_file', type=str, default='/home/renyongpeng/dataset/places128/train_rtv',
                        help='the path of rtv img')
    parser.add_argument('--mask_file', type=str, default='', help='the path of mask')
    parser.add_argument('--save_dir', type=str, default='checkpoint', help='the dir of checkpoint')
    parser.add_argument('--save_dir_rtv', type=str, default='checkpoint_rtv', help='the dir of checkpoint')
    parser.add_argument('--ckpt_iter', type=str, default='160000', help='which iter of ckpt file to be loaded')
    parser.add_argument('--ckpt_iter_rtv', type=str, default='200000', help='which iter of ckpt file to be loaded')
    parser.add_argument('--experiment_name', type=str, default='Res_Outpainting', help='the name of experiment')
    parser.add_argument('--fine_size', type=int, default=128, help='the size of fine img')
    parser.add_argument('--batch_size', type=int, default=14, help='the batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='the batch size')
    parser.add_argument('--gpu_ids', type=str, default='2', help='')

    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='initial learning rate')
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')

    parser.add_argument('--lambda_G', type=float, default=1.0, help='weight of adversarial loss  for G')
    parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight of reconstruction loss')
    parser.add_argument('--lambda_perc', type=float, default=0.1, help='weight of perceptual loss')
    parser.add_argument('--lambda_style', type=float, default=250.0, help='weight of style loss')

    parser.add_argument('--present_iter', type=int, default=1, help='training iteration of presenting loss ')
    parser.add_argument('--display_iter', type=int, default=1000, help='training iteration of displaying image results')
    parser.add_argument('--save_iter', type=int, default=10000, help='training iteration of saving ckpt')
    parser.add_argument('--epoch_count', type=int, default=1, help='')
    parser.add_argument('--niter', type=int, default=5000000, help='')
    parser.add_argument('--niter_decay', type=int, default=0, help='')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

    opt.train = True
    opt.train_rtv = False

    writer = SummaryWriter(log_dir=opt.save_dir)

    # create dataset
    dataset = Places2(opt)
    # create dataloader
    loader = torch.utils.data.DataLoader(dataset, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # create model
    model = OutpaintingModel(opt)
    model_s = Model_S(opt)
    model_s.eval()

    total_iter = 0

    train_result = os.path.join(opt.save_dir, 'train_result')
    if not os.path.exists(train_result):
        os.mkdir(train_result)

    # start to train
    iter = 0
    for i in range(opt.niter + opt.niter_decay):

        for _, data in enumerate(loader):
            imgs, rtv_imgs, masks, img_names = data

            model_s.set_input(rtv_imgs, masks)
            fea_s, rtv_gt, masked_rtv_imgs = model_s.get_fea_s()

            model.set_input(imgs, masks)
            model.optimize_parameters(fea_s)

            iter += 1

            if iter % opt.present_iter == 0:
                loss = model.get_loss()
                writer.add_scalar('loss_D', loss[0].data, iter)
                writer.add_scalar('fake_loss_G', loss[1].data, iter)
                writer.add_scalar('rec_loss', loss[2].data, iter)
                writer.add_scalar('perc_loss', loss[3], iter)
                writer.add_scalar('style_loss', loss[4], iter)

                print('[iter {:d}] loss_D:{:.4f}, fake_loss_G:{:.4f}, rec_loss:{:.4f}, perc_loss:{:.4f}, style_loss:{:.4f}'.format(iter, loss[0], loss[1], loss[2], loss[3], loss[4]))
                # print('[iter {:d}] loss_D:{:.4f}, fake_loss_G:{:.4f}, rec_loss:{:.4f}'.format(iter, loss[0], loss[1], loss[2]))

            if iter % opt.display_iter == 0:
                img_gt, masked_img, img_gen, img_s = model.get_result()
                out = torch.cat(
                    [rtv_gt[0:8], img_gt[0:8], masked_rtv_imgs[0:8], masked_img[0:8], img_s[0:8], img_gen[0:8]], 0)
                grid = torchvision.utils.make_grid(out)
                # writer.add_image('iter_(%d)' % (total_iter+1), grid, total_iter+1)
                grid_name = os.path.join(train_result, 'iter_{:d}.png'.format(iter))
                torchvision.utils.save_image(grid, grid_name)

            if iter % opt.save_iter == 0:
                model.save_network(iter)

        model.update_learning_rate()
