import tensorflow as tf
import argparse
from tools.utils import *
import os
from tqdm import tqdm
from glob import glob
import time
import numpy as np
from net.generator import Generator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    desc = "AnimeGANv2"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_dir', type=str, default='save_model/' + 'generated',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--test_dir', type=str, default='dataset/test/t',
                        help='Directory name of test photos')
    parser.add_argument('--save_dir', type=str, default='Shinkai/t',
                        help='what style you want to get')
    parser.add_argument('--if_adjust_brightness', type=bool, default=True,
                        help='adjust brightness by the real photo')

    """checking arguments"""

    return parser.parse_args()


def load_model(model_dir):
    model = tf.saved_model.load(model_dir)
    return model


def test(model_dir, style_name, test_dir, if_adjust_brightness, img_size=[256, 256]):
    # tf.reset_default_graph()
    result_dir = 'results/' + style_name
    check_folder(result_dir)
    test_files = glob('{}/*.*'.format(test_dir))

    test_generated = load_model(model_dir)

    # stats_graph(tf.get_default_graph())

    # print('Processing image: ' + sample_file)
    sample_file = 'dataset/test/real/19.jpg'
    sample_image = np.asarray(load_test_data(sample_file, img_size))
    image_path = os.path.join(result_dir, '{0}'.format(os.path.basename(sample_file)))
    fake_img = test_generated(sample_image)
    if if_adjust_brightness:
        save_images(fake_img, image_path, sample_file)
    else:
        save_images(fake_img, image_path, None)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
        tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
        # 或者也可以设置GPU显存为固定使用量(例如：4G)
        # tf.config.experimental.set_virtual_device_configuration(gpu0,
        #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        tf.config.set_visible_devices([gpu0], "GPU")

    arg = parse_args()
    print(arg.model_dir)
    test(arg.model_dir, arg.save_dir, arg.test_dir, arg.if_adjust_brightness)
