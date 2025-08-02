import os
import pickle
import argparse

from tensorflow import Module
from tensorflow import Variable
from tensorflow import Tensor
from tensorflow import math as tf_math
from tensorflow import concat as tf_concat
from tensorflow import split as tf_split
from tensorflow import size
from tensorflow.keras import Model as TF_Module
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.saving import register_keras_serializable

def _causal_conv_stack(dilations, out_channels, kernel_size):
    """
    Create stack of dilated convolutional layers, outlined in WaveNet paper:
    https://arxiv.org/pdf/1609.03499.pdf
    """
    moduleList = \
        [
            Conv1D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding="causal",
                dilation_rate=d,
                groups=1,
                use_bias=True,
            )
            for d in dilations
        ]
    return moduleList

class WaveNetTF(Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, **kwargs):
        super().__init__(**kwargs)
        dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
        internal_channels = int(num_channels * 2)
        self.hidden = _causal_conv_stack(dilations, internal_channels, kernel_size)
        self.residuals = _causal_conv_stack(dilations, num_channels, 1)
        self.input_layer = Conv1D(
            filters=num_channels,
            kernel_size=1,
            padding="causal",
        )

        self.linear_mix = Conv1D(
            #in_channels=num_channels * dilation_depth * num_repeat,
            filters=1, #out_channels
            kernel_size=1,
        )

        self.num_channels = num_channels

    def __call__(self, x):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            # I"M ALMOST CERTAIN THIS COMMENT IS WRONG OR AT LEAST MISLEAD ME
            # It looks like you're supposed to split on the 0th dimension, but the pytorch
            # code splits on the 2nd (their 1st)
            # gated activation
            #   split (32,16,3) into two (16,16,3) for tanh and sigm calculations
            out_hidden_split = tf_split(out_hidden,
                                      num_or_size_splits=self.num_channels,
                                      axis=2)
            out = tf_math.tanh(out_hidden_split[0]) * tf_math.sigmoid(out_hidden_split[1])
            skips.append(out)

            out = residual(out)
            outSize = out.get_shape()[1]
            out = out + x[:, -outSize :, :]

        # modified "postprocess" step:
        outSize = out.get_shape()[1]
        out = tf_concat([s[:, -outSize :, :] for s in skips], axis=2)
        out = self.linear_mix(out)
        return out

@register_keras_serializable()
def error_to_signal(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    diff = y - y_pred
    diffPow = tf_math.pow(diff, 2)
    # sum across rows
    diffSum = tf_math.reduce_sum(diffPow, 1)

    yPow = tf_math.pow(y, 2)
    ySum = tf_math.reduce_sum(yPow, 1)
    return diffSum / (ySum + 1e-10)

def pre_emphasis_filter(x, coeff=0.95):
    concat = tf_concat((x[:, 0:1, :], x[:, 1:, :] - coeff * x[:, :-1, :]), axis=1)
    return concat

@register_keras_serializable()
class PedalNetTF(TF_Module):
    def __init__(self, hparams, **kwargs):
        #kwargs['hparams'] = hparams
        super(PedalNetTF, self).__init__(**kwargs)
        self.wavenet = WaveNetTF(
            num_channels=hparams.num_channels,
            dilation_depth=hparams.dilation_depth,
            num_repeat=hparams.num_repeat,
            kernel_size=hparams.kernel_size,
        )
        self._hparams = hparams

    def call(self, inputs, training=False):
        return self.wavenet.__call__(inputs)

    def get_config(self):
        # TODO: hparams is some object, have to figure out how to return a dict
        return vars(self._hparams) 

    @classmethod
    def from_config(cls, config):
        hparams = argparse.Namespace(**config)
        return cls(hparams)

    def prepare_data(self):
        data = pickle.load(open(os.path.dirname(self._hparams.model) + "/data.pickle", "rb"))
        self.x_train = data["x_train"]
        self.y_train = data["y_train"]
        self.x_valid = data["x_valid"]
        self.y_valid = data["y_valid"]
        self.x_test = data["x_test"]
        self.y_test = data["y_test"]

    def optimizer(self):
        return Adam(self._hparams.learning_rate)
