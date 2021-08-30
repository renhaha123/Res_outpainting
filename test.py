import torch
import argparse
from data.dataset import Places2
import torchvision
from models.model import OutpaintingModel
from models.model_S import Model_S
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', type=str, default='/home/renyongpeng/dataset/places128/test',
                        help='the path of img')
    parser.add_argument('--rtv_img_file', type=str, default='/home/renyongpeng/dataset/places128/test_rtv',
                        help='the path of rtv img')
    parser.add_argument('--mask_file', type=str, default='', help='the path of mask')
    parser.add_argument('--save_dir', type=str, default='checkpoint', help='the dir of checkpoint')
    parser.add_argument('--save_dir_rtv', type=str, default='checkpoint_rtv', help='the dir of checkpoint')
    parser.add_argument('--ckpt_iter', type=str, default='120000', help='which iter of ckpt file to be loaded')
    parser.add_argument('--ckpt_iter_rtv', type=str, default='200000', help='which iter of ckpt file to be loaded')
    parser.add_argument('--test_save', type=str, default='test_result', help='which iter of ckpt file to be loaded')
    parser.add_argument('--experiment_name', type=str, default='Res_Outpainting', help='the name of experiment')
    parser.add_argument('--fine_size', type=int, default=128, help='the size of fine img')
    parser.add_argument('--batch_size', type=int, default=12, help='the batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='the batch size')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: 0 or 0,1 or 0,1,2')

    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

    opt.train = False
    opt.train_rtv = False

    # create dataset
    dataset = Places2(opt)
    # create dataloader
    loader = torch.utils.data.DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # create model
    model = OutpaintingModel(opt)
    model.eval()

    model_s = Model_S(opt)
    model_s.eval()

    test_result = os.path.join(opt.test_save)
    if not os.path.exists(test_result):
        os.mkdir(test_result)

    # start to test
    for _, data in enumerate(loader):
        imgs, rtv_imgs, masks, img_names = data

        model_s.set_input(rtv_imgs, masks)
        fea_s, rtv_gt, masked_rtv_imgs = model_s.get_fea_s()

        model.set_input(imgs, masks)
        results = model.test(fea_s)

        # save images
        results = (results + 1.0) / 2.0
        for i in range(results.size()[0]):
            img = results[i]
            img_name = os.path.join(opt.test_save, img_names[i])
            print('Saving the {:s}'.format(img_name))
            torchvision.utils.save_image(img, img_name)

