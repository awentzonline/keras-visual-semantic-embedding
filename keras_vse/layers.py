from keras import backend as K
from keras.layers import Layer


class L2Normalize(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        return super(L2Normalize, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.l2_normalize(x, axis=self.axis)

    def get_config(self):
        base_config = super(L2Normalize, self).get_config()
        base_config.update(dict(axis=self.axis))
        return base_config
