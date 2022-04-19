import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, LayerNormalization
from tensorflow_addons.layers import SpectralNormalization


class Conv(Layer):
    def __init__(self, fiflters, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False):
        super(Conv, self).__init__()
        if (kernel - stride) % 2 == 0:
            self.pad_top = pad
            self.pad_bottom = pad
            self.pad_left = pad
            self.pad_right = pad

        else:
            self.pad_top = pad
            self.pad_bottom = kernel - stride - self.pad_top
            self.pad_left = pad
            self.pad_right = kernel - stride - self.pad_left

        self.sn = sn
        self.fiflters = fiflters
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.stride = stride
        self.kernel = kernel
        self.spectral_norm = SpectralNormalization(Conv2D(filters=self.fiflters, kernel_size=self.kernel,
                                                          kernel_initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                          stddev=0.02),
                                                          padding='VALID', use_bias=self.use_bias,
                                                          strides=self.stride,
                                                          bias_initializer=tf.constant_initializer(0.0)))
        self.conv = Conv2D(filters=self.fiflters,
                           kernel_size=self.kernel,
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                           strides=self.stride, use_bias=self.use_bias)

    def build(self, input_shape):
        super(Conv, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        if self.pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [self.pad_top, self.pad_bottom], [self.pad_left, self.pad_right], [0, 0]])
        if self.pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [self.pad_top, self.pad_bottom], [self.pad_left, self.pad_right], [0, 0]],
                       mode='REFLECT')

        if self.sn:
            x = self.spectral_norm(x)
        else:
            x = self.conv(x)
        return x


class Discriminator(Model):
    def __init__(self, ch, n_dis, sn):
        super(Discriminator, self).__init__(name="Discriminator")
        channel = ch // 2

        self.conv1 = Conv(fiflters=channel, kernel=3, stride=1, pad=1, use_bias=False, sn=sn)
        self.leaky_relu1 = LeakyReLU(0.2)

        self.listLayers = []
        for i in range(1, n_dis):
            self.listLayers.append(Conv(channel * 2, kernel=3, stride=2, pad=1, use_bias=False, sn=sn))

            self.listLayers.append(LeakyReLU(alpha=0.2))

            self.listLayers.append(Conv(channel * 4, kernel=3, stride=1, pad=1, use_bias=False, sn=sn))
            self.listLayers.append(LayerNormalization(axis=-1, center=True, scale=True))
            self.listLayers.append(LeakyReLU(alpha=0.2))

            channel = channel * 2

        self.conv2 = Conv(channel * 2, kernel=3, stride=1, pad=1, use_bias=False, sn=sn)
        self.layer_norm2 = LayerNormalization(axis=-1, center=True, scale=True)
        self.leaky_relu2 = LeakyReLU(alpha=0.2)

        self.conv3 = Conv(fiflters=1, kernel=3, stride=1, pad=1, use_bias=False, sn=sn)

    def build(self, input_shape):
        super(Discriminator, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.leaky_relu1(x)

        for layer in self.listLayers:
            x = layer(x)

        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        return x
