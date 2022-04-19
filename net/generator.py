import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, LeakyReLU, DepthwiseConv2D, LayerNormalization
from tensorflow.keras.models import Model


class CusConv2D(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='VALID', use_bias=None):
        super(CusConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding=padding, use_bias=use_bias,
                           kernel_initializer=tf.keras.initializers.VarianceScaling(),
                           bias_initializer=tf.constant_initializer(0.0))

    def build(self, input_shape):
        super(CusConv2D, self).build(input_shape)

    def call(self, inputs):
        if self.kernel_size == 3 and self.strides == 1:
            inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        if self.kernel_size == 7 and self.strides == 1:
            inputs = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
        if self.strides == 2:
            inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="REFLECT")

        x = self.conv(inputs)
        return x


class Conv2DNormLReLU(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='VALID', use_bias=None):
        super(Conv2DNormLReLU, self).__init__()
        self.cus_conv2d = CusConv2D(filters, kernel_size, strides, padding, use_bias)
        self.leaky_relu = LeakyReLU(alpha=0.2)

    def build(self, input_shape):
        super(Conv2DNormLReLU, self).build(input_shape)

    def call(self, inputs):
        x = self.cus_conv2d(inputs)
        x = self.leaky_relu(x)
        return x


class DwiseConv2D(Layer):
    def __init__(self, kernel_size=3, strides=1, padding='VALID', channel_multiplier=1, use_bias=True):
        super(DwiseConv2D, self).__init__()
        self.depthwise_conv2d = DepthwiseConv2D(kernel_size=kernel_size, padding=padding,
                                                use_bias=use_bias,
                                                depthwise_initializer=tf.keras.initializers.VarianceScaling(),
                                                strides=strides, bias_initializer=tf.constant_initializer(0.0),
                                                depth_multiplier=channel_multiplier)

    def build(self, input_shape):
        super(DwiseConv2D, self).build(input_shape)

    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        x = self.depthwise_conv2d(x)
        return x


class InvertedResidualBlock(Layer):
    def __init__(self, output_dim, stride, bias=None, expansion_ratio=2):
        super(InvertedResidualBlock, self).__init__()
        self.output_dim = output_dim
        self.stride = stride
        self.bias = bias
        self.expansion_ratio = expansion_ratio
        self.conv2d_norm_lrelu = None
        self.dwise_conv2d = DwiseConv2D()
        self.layer_norm = LayerNormalization()
        self.leaky_relu = LeakyReLU(alpha=0.2)
        self.cus_conv2d = CusConv2D(filters=self.output_dim, kernel_size=1)
        self.layer_norm2 = LayerNormalization()

    def build(self, input_shape):
        input_channel = int(input_shape[-1])
        bottleneck_dim = round(self.expansion_ratio * input_channel)
        self.conv2d_norm_lrelu = Conv2DNormLReLU(bottleneck_dim, kernel_size=1, use_bias=self.bias)
        super(InvertedResidualBlock, self).build(input_shape)

    def call(self, inputs):
        # pw
        net = self.conv2d_norm_lrelu(inputs)

        # dw
        net = self.dwise_conv2d(net)
        net = self.layer_norm(net)
        net = self.leaky_relu(net)

        # pw & linear
        net = self.cus_conv2d(net)
        net = self.layer_norm2(net)

        # element wise add, only for stride==1
        if (int(inputs.get_shape().as_list()[-1]) == self.output_dim) and self.stride == 1:
            net = inputs + net

        return net


class Unsample(Layer):
    def __init__(self, filters, kernel_size=3):
        super(Unsample, self).__init__()
        self.conv2d_norm_lrelu = Conv2DNormLReLU(filters, kernel_size=kernel_size)

    def build(self, input_shape):
        super(Unsample, self).build(input_shape)

    def call(self, inputs):
        new_H, new_W = 2 * tf.shape(inputs)[1], 2 * tf.shape(inputs)[2]
        inputs = tf.image.resize(inputs, [new_H, new_W])
        return self.conv2d_norm_lrelu(inputs)


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__(name='Generator')

        self.A_block = [
            Conv2DNormLReLU(filters=32, kernel_size=7),
            Conv2DNormLReLU(filters=64, strides=2),
            Conv2DNormLReLU(filters=64),
        ]

        self.B_block = [
            Conv2DNormLReLU(filters=128, strides=2),
            Conv2DNormLReLU(filters=128),
        ]

        self.C_block = [
            Conv2DNormLReLU(filters=128),
            InvertedResidualBlock(expansion_ratio=2, output_dim=256, stride=1),
            InvertedResidualBlock(expansion_ratio=2, output_dim=256, stride=1),
            InvertedResidualBlock(expansion_ratio=2, output_dim=256, stride=1),
            InvertedResidualBlock(expansion_ratio=2, output_dim=256, stride=1),
            Conv2DNormLReLU(filters=128),
        ]

        self.D_block = [
            Unsample(filters=128, kernel_size=3),
            Conv2DNormLReLU(filters=128),
        ]

        self.E_block = [
            Unsample(filters=64, kernel_size=3),
            Conv2DNormLReLU(filters=64),
            Conv2DNormLReLU(filters=32, kernel_size=7),
        ]

        self.cus_conv = CusConv2D(filters=3, kernel_size=1, strides=1)

    def build(self, input_shape):
        super(Generator, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for layer in self.A_block:
            x = layer(x)
        for layer in self.B_block:
            x = layer(x)
        for layer in self.C_block:
            x = layer(x)
        for layer in self.D_block:
            x = layer(x)
        for layer in self.E_block:
            x = layer(x)
        x = self.cus_conv(x)
        fake = tf.tanh(x)
        return fake
