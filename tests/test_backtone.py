from unittest import TestCase

import numpy as np

from net.backtone import VGGCaffePreTrained
from PIL import Image
import tensorflow as tf

from tools.utils import preprocessing


class TestVGGCaffePreTrained(TestCase):
    model = VGGCaffePreTrained()

    image = Image.open("../dataset/test/test_photo256/0.png")
    image = image.resize((224, 224))
    np_img = np.array(image).astype('float32')
    np_img = preprocessing(np_img, size=(224, 224))

    img = tf.convert_to_tensor(np_img)
    img = tf.expand_dims(img, axis=0)
    feat = model(img)

    print(feat.shape)

