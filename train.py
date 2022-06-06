from AnimeGANv2 import AnimeGANv2
import tensorflow as tf
import argparse
from tools.utils import *
import os
import wandb

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


"""parsing and configuration"""


def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config_path', type=str, help='hyper params config path', required=True)
    parser.add_argument('--dataset', type=str, default='Hayao', help='dataset_name')
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')
    parser.add_argument('--img_size', type=list, default=[256, 256], help='The size of image: H and W')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')
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
    gan.train()
    print(" [*] Training finished!")


if __name__ == '__main__':
    main()
