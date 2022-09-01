import unittest
import tensorflow as tf


class MyTestCase(unittest.TestCase):
    def test_something(self):
        inputs = tf.keras.Input([256, 256, 3])
        model = tf.keras.applications.MobileNetV2(
            include_top=False,
            alpha=1.3,
            weights='imagenet',
            input_tensor=inputs,
            pooling=None,
            classes=1000)
        print(model.summary())
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
