import tensorflow as tf


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


def discriminator_loss(real, gray, fake, real_blur):
    real_loss = tf.reduce_mean(tf.square(real - 1.0))
    gray_loss = tf.reduce_mean(tf.square(gray))
    fake_loss = tf.reduce_mean(tf.square(fake))
    real_blur_loss = tf.reduce_mean(tf.square(real_blur))
    return real_loss, fake_loss, gray_loss, real_blur_loss


def generator_loss(fake):
    fake_loss = tf.reduce_mean(tf.square(fake - 1.0))
    return fake_loss


def gram(x):
    result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    input_shape = tf.shape(x)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


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
    anime_feature_map = pre_train_model(anime[:fake_feature_map.shape[0]], training=False)
    c_loss = L1_loss(real_feature_map, fake_feature_map)
    s_loss = style_loss(anime_feature_map, fake_feature_map)
    return c_loss, s_loss


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
    return tf.image.rgb_to_yuv(rgb)
