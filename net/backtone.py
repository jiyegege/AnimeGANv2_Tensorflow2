import os

import keras
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Layer, Conv2D, MaxPool2D

VGG_MEAN = [103.939, 116.779, 123.68]


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
        mean: tf.Tensor = tf.constant(VGG_MEAN, dtype=tf.float32)
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


class VGG19Conv4(Model):
    def __init__(self, weights_path: str = os.path.dirname(os.path.abspath(__file__)) + '/../models/vgg19.npy'):
        super().__init__()
        try:
            self.data_dict: dict = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
            print("npy file loaded")
        except FileNotFoundError as e:
            print("weights_path:", weights_path,
                  'does not exits!, if you want to training must download pretrained weights')
        conv1_1_filter = self.get_conv_filter('conv1_1')
        conv1_1_filter_shape = conv1_1_filter.shape
        conv1_1_biases = self.get_bias('conv1_1')
        self.conv1_1 = Conv2D(conv1_1_filter_shape[3], kernel_size=(conv1_1_filter_shape[0], conv1_1_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv1_1', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv1_1_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv1_1_filter))
        conv1_2_filter = self.get_conv_filter('conv1_2')
        conv1_2_filter_shape = conv1_2_filter.shape
        conv1_2_biases = self.get_bias('conv1_2')
        self.conv1_2 = Conv2D(conv1_2_filter_shape[3], kernel_size=(conv1_2_filter_shape[0], conv1_2_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv1_2', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv1_2_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv1_2_filter))
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        conv2_1_filter = self.get_conv_filter('conv2_1')
        conv2_1_filter_shape = conv2_1_filter.shape
        conv2_1_biases = self.get_bias('conv2_1')
        self.conv2_1 = Conv2D(conv2_1_filter_shape[3], kernel_size=(conv2_1_filter_shape[0], conv2_1_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv2_1', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv2_1_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv2_1_filter))
        conv2_2_filter = self.get_conv_filter('conv2_2')
        conv2_2_filter_shape = conv2_2_filter.shape
        conv2_2_biases = self.get_bias('conv2_2')
        self.conv2_2 = Conv2D(conv2_2_filter_shape[3], kernel_size=(conv2_2_filter_shape[0], conv2_2_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv2_2', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv2_2_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv2_2_filter))
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        conv3_1_filter = self.get_conv_filter('conv3_1')
        conv3_1_filter_shape = conv3_1_filter.shape
        conv3_1_biases = self.get_bias('conv3_1')
        self.conv3_1 = Conv2D(conv3_1_filter_shape[3], kernel_size=(conv3_1_filter_shape[0], conv3_1_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv3_1', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv3_1_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv3_1_filter))
        conv3_2_filter = self.get_conv_filter('conv3_2')
        conv3_2_filter_shape = conv3_2_filter.shape
        conv3_2_biases = self.get_bias('conv3_2')
        self.conv3_2 = Conv2D(conv3_2_filter_shape[3], kernel_size=(conv3_2_filter_shape[0], conv3_2_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv3_2', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv3_2_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv3_2_filter))
        conv3_3_filter = self.get_conv_filter('conv3_3')
        conv3_3_filter_shape = conv3_3_filter.shape
        conv3_3_biases = self.get_bias('conv3_3')
        self.conv3_3 = Conv2D(conv3_3_filter_shape[3], kernel_size=(conv3_3_filter_shape[0], conv3_3_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv3_3', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv3_3_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv3_3_filter))
        conv3_4_filter = self.get_conv_filter('conv3_4')
        conv3_4_filter_shape = conv3_4_filter.shape
        conv3_4_biases = self.get_bias('conv3_4')
        self.conv3_4 = Conv2D(conv3_4_filter_shape[3], kernel_size=(conv3_4_filter_shape[0], conv3_4_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv3_4', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv3_4_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv3_4_filter))
        self.pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        conv4_1_filter = self.get_conv_filter('conv4_1')
        conv4_1_filter_shape = conv4_1_filter.shape
        conv4_1_biases = self.get_bias('conv4_1')
        self.conv4_1 = Conv2D(conv4_1_filter_shape[3], kernel_size=(conv4_1_filter_shape[0], conv4_1_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv4_1', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv4_1_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv4_1_filter))
        conv4_2_filter = self.get_conv_filter('conv4_2')
        conv4_2_filter_shape = conv4_2_filter.shape
        conv4_2_biases = self.get_bias('conv4_2')
        self.conv4_2 = Conv2D(conv4_2_filter_shape[3], kernel_size=(conv4_2_filter_shape[0], conv4_2_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv4_2', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv4_2_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv4_2_filter))
        conv4_3_filter = self.get_conv_filter('conv4_3')
        conv4_3_filter_shape = conv4_3_filter.shape
        conv4_3_biases = self.get_bias('conv4_3')
        self.conv4_3 = Conv2D(conv4_3_filter_shape[3], kernel_size=(conv4_3_filter_shape[0], conv4_3_filter_shape[1]),
                              padding='same', activation='relu',
                              strides=(1, 1), name='conv4_3', use_bias=True,
                              bias_initializer=tf.keras.initializers.Constant(conv4_3_biases),
                              kernel_initializer=tf.keras.initializers.Constant(conv4_3_filter))

        conv4_4_filter = self.get_conv_filter('conv4_4')
        conv4_4_filter_shape = conv4_4_filter.shape
        conv4_4_biases = self.get_bias('conv4_4')
        self.conv4_4_no_activation = Conv2D(conv4_4_filter_shape[3],
                                            kernel_size=(conv4_4_filter_shape[0], conv4_4_filter_shape[1]),
                                            padding='same', activation=None,
                                            strides=(1, 1), name='conv4_4', use_bias=True,
                                            bias_initializer=tf.keras.initializers.Constant(conv4_4_biases),
                                            kernel_initializer=tf.keras.initializers.Constant(conv4_4_filter))

    def call(self, inputs, training=False, mask=None):
        rgb_scaled = ((inputs + 1) / 2) * 255.0  # [-1, 1] ~ [0, 255]

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        inputs = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                                           green - VGG_MEAN[1],
                                           red - VGG_MEAN[2]])
        net = self.conv1_1(inputs)
        net = self.conv1_2(net)
        net = self.pool1(net)

        net = self.conv2_1(net)
        net = self.conv2_2(net)
        net = self.pool2(net)

        net = self.conv3_1(net)
        net = self.conv3_2(net)
        net = self.conv3_3(net)
        net = self.conv3_4(net)
        net = self.pool3(net)

        net = self.conv4_1(net)
        net = self.conv4_2(net)
        net = self.conv4_3(net)
        net = self.conv4_4_no_activation(net)
        return net

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")
