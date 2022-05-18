import tensorflow as tf
import wandb
from tensorflow import keras
from tensorflow.keras.layers import Conv2D

from tensorflow_addons.layers import SpectralNormalization


def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def Huber_loss(x, y):
    h = tf.keras.losses.Huber()
    return h(x, y)


def discriminator_loss(loss_func, real, gray, fake, real_blur, step, writer, real_loss_weight,
                       fake_loss_weight, gray_loss_weight, real_blur_loss_weight):
    real_loss = 0
    gray_loss = 0
    fake_loss = 0
    real_blur_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        real_loss = -tf.reduce_mean(real)
        gray_loss = tf.reduce_mean(gray)
        fake_loss = tf.reduce_mean(fake)
        real_blur_loss = tf.reduce_mean(real_blur)

    if loss_func == 'lsgan':
        real_loss = tf.reduce_mean(tf.square(real - 1.0))
        gray_loss = tf.reduce_mean(tf.square(gray))
        fake_loss = tf.reduce_mean(tf.square(fake))
        real_blur_loss = tf.reduce_mean(tf.square(real_blur))

    if loss_func == 'gan' or loss_func == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        gray_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(gray), logits=gray))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
        real_blur_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_blur), logits=real_blur))

    if loss_func == 'hinge':
        real_loss = tf.reduce_mean(relu(1.0 - real))
        gray_loss = tf.reduce_mean(relu(1.0 + gray))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))
        real_blur_loss = tf.reduce_mean(relu(1.0 + real_blur))

    # for Hayao : 1.2, 1.2, 1.2, 0.8
    # for Paprika : 1.0, 1.0, 1.0, 0.005
    # for Shinkai: 1.7, 1.7, 1.7, 1.0
    loss = real_loss_weight * real_loss + fake_loss_weight * fake_loss \
           + gray_loss_weight * gray_loss + real_blur_loss_weight * real_blur_loss

    # wandb.log("Discriminator_real_loss", real_loss.numpy(), step=step)
    # wandb.log("Discriminator_fake_loss", fake_loss.numpy(), step=step)
    # wandb.log("Discriminator_gray_loss", gray_loss.numpy(), step=step)
    # wandb.log("Discriminator_real_blur_loss", real_blur_loss.numpy(), step=step)
    with writer.as_default(step=step):
        """" Summary """
        tf.summary.scalar("Discriminator_real_loss", real_loss)
        tf.summary.scalar("Discriminator_fake_loss", fake_loss)
        tf.summary.scalar("Discriminator_gray_loss", gray_loss)
        tf.summary.scalar("Discriminator_real_blur_loss", real_blur_loss)

    return loss


def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

    if loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss


def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)


def con_loss(pre_train_model, real, fake):
    real_feature_map = pre_train_model(real, training=False)

    fake_feature_map = pre_train_model(fake, training=False)

    loss = L1_loss(real_feature_map, fake_feature_map)

    return loss


def style_loss(style, fake):
    return L1_loss(gram(style), gram(fake))


def con_sty_loss(pre_train_model, real, anime, fake):
    real_feature_map = pre_train_model(real, training=False)

    fake_feature_map = pre_train_model(fake, training=False)

    anime_feature_map = pre_train_model(anime, training=False)

    c_loss = L1_loss(real_feature_map, fake_feature_map)
    s_loss = style_loss(anime_feature_map, fake_feature_map)

    return c_loss, s_loss


def local_variables_init():
    inputs = tf.keras.Input([256, 256, 3])
    model = tf.keras.applications.MobileNetV2(
        include_top=False,
        alpha=1.3,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None,
        classes=1000)
    p_model: tf.keras.Model = tf.keras.Model(
        inputs,
        model.get_layer('block_6_expand').output)
    p_model.trainable = False
    return p_model


def color_loss(con, fake):
    con = rgb2yuv(con)
    fake = rgb2yuv(fake)

    return L1_loss(con[:, :, :, 0], fake[:, :, :, 0]) + Huber_loss(con[:, :, :, 1], fake[:, :, :, 1]) + Huber_loss(
        con[:, :, :, 2], fake[:, :, :, 2])


def total_variation_loss(inputs):
    """
    A smooth loss in fact. Like the smooth prior in MRF.
    V(y) = || y_{n+1} - y_n ||_2
    """
    dh = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dw = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    size_dh = tf.size(dh, out_type=tf.float32)
    size_dw = tf.size(dw, out_type=tf.float32)
    return tf.nn.l2_loss(dh) / size_dh + tf.nn.l2_loss(dw) / size_dw


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb = (rgb + 1.0) / 2.0
    # rgb2yuv_filter = tf.constant([[[[0.299, -0.169, 0.499],
    #                                 [0.587, -0.331, -0.418],
    #                                 [0.114, 0.499, -0.0813]]]])
    # rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
    # temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    # temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    # return temp
    return tf.image.rgb_to_yuv(rgb)
