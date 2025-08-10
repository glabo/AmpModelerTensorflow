import os
import pickle
import argparse
import traceback

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

from tensorflow import convert_to_tensor

def _causal_conv_stack(dilations, out_channels, kernel_size, name):
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
                name=name + "_" + str(i)
            )
            for i, d in enumerate(dilations)
        ]
    return moduleList

@register_keras_serializable()
def error_to_signal(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y = y[:, -y_pred.get_shape()[1] :, :]
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    diff = y - y_pred
    diffPow = tf_math.square(diff)
    diffSum = tf_math.reduce_sum(diffPow, 1)

    yPow = tf_math.square(y)
    ySum = tf_math.reduce_sum(yPow, 1)

    err = diffSum / (ySum + 1e-10)
    mean = tf_math.reduce_mean(err) 
    return mean

def pre_emphasis_filter(x, coeff=0.95):
    concat = tf_concat((x[:, 0:1, :], x[:, 1:, :] - coeff * x[:, :-1, :]), axis=1)
    return concat

@register_keras_serializable()
class PedalNetTF(TF_Module):
    def __init__(self, hparams, **kwargs):
        super(PedalNetTF, self).__init__(**kwargs)
        dilations = [2 ** d for d in range(hparams.dilation_depth)] * hparams.num_repeat
        internal_channels = int(hparams.num_channels * 2)
        self.hidden = _causal_conv_stack(dilations, internal_channels, hparams.kernel_size, "hidden")
        self.residuals = _causal_conv_stack(dilations, hparams.num_channels, 1, "residual")
        self.input_layer = Conv1D(
            filters=hparams.num_channels,
            kernel_size=1,
            padding="causal",
            name='input_layer',
        )

        self.linear_mix = Conv1D(
            filters=1, #out_channels
            kernel_size=1,
            name='linear_mix',
        )

        self.num_channels = hparams.num_channels
        self._hparams = hparams

    def call(self, inputs):
        skips = []
        layer_output = self.input_layer(inputs)

        for hidden, residual in zip(self.hidden, self.residuals):
            current_layer_input = layer_output
            residual_input = current_layer_input 
            out_hidden = hidden(current_layer_input)

            # gated activation
            out_hidden_split = tf_split(out_hidden,
                                      num_or_size_splits=self.num_channels,
                                      axis=2)
            gated_output = tf_math.tanh(out_hidden_split[0]) * tf_math.sigmoid(out_hidden_split[1])
            skips.append(gated_output)

            residual_output = residual(gated_output)
            outSize = residual_output.get_shape()[1]
            layer_output = residual_output + residual_input[:, -outSize :, :]

        # modified "postprocess" step:
        outSize = layer_output.get_shape()[1]
        skip_concat = tf_concat([s[:, -outSize :, :] for s in skips], axis=2)
        out = self.linear_mix(skip_concat)
        return out

    def get_config(self):
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
        return Adam(learning_rate=self._hparams.learning_rate)

# class WaveNetTF(Module):
#     def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, **kwargs):
#         super().__init__(**kwargs)
#         dilations = [2 ** d for d in range(dilation_depth)] * num_repeat
#         internal_channels = int(num_channels * 2)
#         self.hidden = _causal_conv_stack(dilations, internal_channels, kernel_size)
#         self.residuals = _causal_conv_stack(dilations, num_channels, 1)
#         self.input_layer = Conv1D(
#             filters=num_channels,
#             kernel_size=1,
#             padding="causal",
#         )
#
#         self.linear_mix = Conv1D(
#             #in_channels=num_channels * dilation_depth * num_repeat,
#             filters=1, #out_channels
#             kernel_size=1,
#         )
#
#         self.num_channels = num_channels
#
#     def print_layer_conf_and_weights(self, layer):
#         print(layer.name, "  config: ", layer.get_config(), "  weights: ", layer.get_weights())
#
#     def print_model_weights(self):
#         print("Layer weights:")
#         self.print_layer_conf_and_weights(self.input_layer)
#         for hidden, residual in zip(self.hidden, self.residuals):
#             print("hidden: ", hidden.get_weights())
#             print("residual: ", residual.get_weights())
#         print("linear_mix: ", self.linear_mix.get_weights())
#
#     def __call__(self, input):
#         skips = []
#         layer_output = self.input_layer(input)
#
#         for hidden, residual in zip(self.hidden, self.residuals):
#             current_layer_input = layer_output
#             residual_input = current_layer_input 
#             out_hidden = hidden(current_layer_input)
#
#             # gated activation
#             out_hidden_split = tf_split(out_hidden,
#                                       num_or_size_splits=self.num_channels,
#                                       axis=2)
#             gated_output = tf_math.tanh(out_hidden_split[0]) * tf_math.sigmoid(out_hidden_split[1])
#             skips.append(gated_output)
#
#             residual_output = residual(gated_output)
#             outSize = residual_output.get_shape()[1]
#             layer_output = residual_output + residual_input[:, -outSize :, :]
#
#         # modified "postprocess" step:
#         outSize = layer_output.get_shape()[1]
#         skip_concat = tf_concat([s[:, -outSize :, :] for s in skips], axis=2)
#         out = self.linear_mix(skip_concat)
#         return out
#
