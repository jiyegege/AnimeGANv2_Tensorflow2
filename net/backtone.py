import os

import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Layer, Conv2D


class VGGCaffePreTrained(Model):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
           'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

    def __init__(self, weights_path: str = os.path.dirname(os.path.abspath(__file__)) + '/../models/vgg19.npy',
                 output_index: int = 26):
        super().__init__()
        try:
            data_dict: dict = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
            self.features = self.make_layers(self.cfg, data_dict)
            del data_dict
        except FileNotFoundError as e:
            print("weights_path:", weights_path,
                  'does not exits!, if you want to training must download pretrained weights')
        self.output_index = output_index
        self.vgg_normalize = None

    def _process(self, x):
        # NOTE 图像范围为[-1~1]，先denormalize到0-1再归一化
        rgb = (x * 0.5 + 0.5) * 255.0
        bgr = tf.reverse(rgb, axis=[-1])
        return self.vgg_normalize(bgr)

    def build(self, input_shape):
        mean: tf.Tensor = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        mean = mean[None, None, None, :]
        self.vgg_normalize = lambda x: x - mean

    def _forward_impl(self, x):
        x = self._process(x)
        # NOTE get output without relu activation
        for i, layer in enumerate(self.features.layers):
            x = layer(x)
            if i == self.output_index:
                break
        return x

    def call(self, x):
        return self._forward_impl(x)

    @staticmethod
    def get_conv_filter(data_dict, name):
        return data_dict[name][0]

    @staticmethod
    def get_bias(data_dict, name):
        return data_dict[name][1]

    @staticmethod
    def get_fc_weight(data_dict, name):
        return data_dict[name][0]

    def make_layers(self, cfg, data_dict, batch_norm=False) -> keras.Sequential:
        layers = []
        block = 1
        number = 1
        for v in cfg:
            if v == 'M':
                layers += [keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')]
                block += 1
                number = 1
            else:
                weight = self.get_conv_filter(data_dict, f'conv{block}_{number}')

                bias = self.get_bias(data_dict, f'conv{block}_{number}')
                conv2d = Conv2D(v, kernel_size=(3, 3), padding='same', use_bias=True,
                                kernel_initializer=tf.keras.initializers.Constant(weight),
                                bias_initializer=tf.keras.initializers.Constant(bias))
                number += 1
                if batch_norm:
                    layers += [conv2d,
                               keras.layers.BatchNormalization(),
                               keras.layers.ReLU()]
                else:
                    layers += [conv2d,
                               keras.layers.ReLU()]
        return keras.Sequential(layers)
