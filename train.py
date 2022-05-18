from AnimeGANv2 import AnimeGANv2
import tensorflow as tf
import argparse
from tools.utils import *
import os
import wandb

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"



"""parsing and configuration"""


def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='Hayao', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=101, help='The number of epochs to run')
    parser.add_argument('--init_epoch', type=int, default=10, help='The number of epochs for weight initialization')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='The size of batch size')  # if light : batch_size = 20
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')

    parser.add_argument('--init_lr', type=float, default=2e-4, help='The learning rate')
    parser.add_argument('--g_lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=4e-5, help='The learning rate')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')

    parser.add_argument('--g_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--d_adv_weight', type=float, default=300.0, help='Weight about GAN')
    parser.add_argument('--con_weight', type=float, default=1.5,
                        help='Weight about VGG19')  # 1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
    # ------ the follow weight used in AnimeGAN
    parser.add_argument('--sty_weight', type=float, default=2.5,
                        help='Weight about style')  # 2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai
    parser.add_argument('--color_weight', type=float, default=10.,
                        help='Weight about color')  # 15. for Hayao, 50. for Paprika, 10. for Shinkai
    parser.add_argument('--tv_weight', type=float, default=1.,
                        help='Weight about tv')  # 1. for Hayao, 0.1 for Paprika, 1. for Shinkai
    # ---------------------------------------------
    parser.add_argument('--training_rate', type=int, default=1, help='training rate about G & D')
    parser.add_argument('--gan_type', type=str, default='lsgan',
                        help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge')

    parser.add_argument('--img_size', type=list, default=[256, 256], help='The size of image: H and W')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    parser.add_argument('--hyperparameters', type=bool, default=False)

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    # gpu_options = tf.GPUOptions(allow_growth=True)

    gan = AnimeGANv2(args)

    if args.hyperparameters:
        sweep_config = {
            'method': 'random',  # grid, random
            'metric': {
                'name': 'Discriminator_fake_loss'
            },
            'parameters': {
                'gan_type': {
                    'values': ['lsgan', 'wgan-gp', 'dragan', 'hinge', 'wgan-lp', 'gan']
                },
                'init_lr': {
                    'values': [2e-4, 2e-5, 2e-6, 3e-4, 3e-5, 3e-6]
                },
                'g_lr': {
                    'values': [2e-5, 2e-6, 2e-7, 3e-5, 3e-6, 3e-7]
                },
                'd_lr': {
                    'values': [4e-5, 4e-6, 4e-7, 5e-5, 5e-6, 5e-7]
                },
                'ld': {
                    'values': [10.0, 11.0, 9.0]
                },
                'g_adv_weight': {
                    'values': [300.0, 150.0, 100.0]
                },
                'd_adv_weight': {
                    'values': [300.0, 150.0, 100.0]
                },
                'con_weight': {
                    'values': [1.5, 2.0, 1.2, 1.5, 1.0]
                },
                'sty_weight': {
                    'values': [2.5, 0.6, 2.0, 1.5, 1.0]
                },
                'color_weight': {
                    'values': [10.0, 50.0, 10.0, 20.0, 30.0]
                },
                'tv_weight': {
                    'values': [1.0, 0.1, 1.2, 1.5, 0.5]
                },
                'real_loss_weight': {
                    'values': [1.2, 1.0, 1.7, 1.5, 0.8]
                },
                'fake_loss_weight': {
                    'values': [1.2, 1.0, 1.7, 1.5, 0.8]
                },
                'gray_loss_weight': {
                    'values': [1.2, 1.0, 1.7, 1.5, 0.8]
                },
                'real_blur_loss_weight': {
                    'values': [0.8, 0.005, 1.0, 0.06, 1.7, 1.2]
                },
                'training_rate': {
                    'values': [1, 2, 4, 6]
                }
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 3
            }
        }

        sweep_id = wandb.sweep(sweep_config, entity="roger_ds", project="AnimeGanV2")
        wandb.agent(sweep_id, gan.train, count=10)
    else:
        gan.train()
    print(" [*] Training finished!")


if __name__ == '__main__':
    main()
