"""
@Author: Conghao Wong
@Date: 2022-09-13 21:18:29
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-06 20:42:09
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from qpid.constant import INPUT_TYPES, PROCESS_TYPES
from qpid.model import transformer
from qpid.silverballers import AgentArgs, BaseAgentModel, BaseAgentStructure
from qpid.utils import POOLING_BEFORE_SAVING


def GraphConv_layer(output_units, activation=None):
    return tf.keras.layers.Dense(output_units, activation)


def GraphConv_func(features, A, output_units=64, activation=None, layer=None):
    dot = tf.matmul(A, features)
    if layer == None:
        res = tf.keras.layers.Dense(output_units, activation)(dot)
    else:
        res = layer(dot)
    return res


class MSNAlphaModel(BaseAgentModel):

    def __init__(self, Args: AgentArgs,
                 as_single_model: bool = True,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Preprocess
        self.set_preprocess(**{PROCESS_TYPES.MOVE: 0})

        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ, INPUT_TYPES.MAP)

        # Layers
        # context feature
        if not POOLING_BEFORE_SAVING:
            self.average_pooling = tf.keras.layers.AveragePooling2D([5, 5],
                                                                    input_shape=[100, 100, 1])

        self.flatten = tf.keras.layers.Flatten()
        self.context_dense1 = tf.keras.layers.Dense(self.args.obs_frames * 64,
                                                    activation=tf.nn.tanh)

        # traj embedding
        self.pos_embedding = tf.keras.layers.Dense(64, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate()

        # trajectory transformer
        self.T1 = transformer.Transformer(num_layers=4,
                                          d_model=128,
                                          num_heads=8,
                                          dff=512,
                                          input_vocab_size=None,
                                          target_vocab_size=None,
                                          pe_input=Args.obs_frames,
                                          pe_target=Args.obs_frames,
                                          include_top=False)

        # transfer GCN
        self.adj_dense2 = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.gcn_transfer = GraphConv_layer(128, tf.nn.tanh)

        # decoder
        self.decoder = tf.keras.layers.Dense(2)

    def call(self, inputs: list[tf.Tensor], training=None, mask=None):
        positions = inputs[0]
        maps = inputs[1]

        # traj embedding, shape == (batch, obs, 64)
        positions_embedding = self.pos_embedding(positions)

        # context feature, shape == (batch, obs, 64)
        if not POOLING_BEFORE_SAVING:
            average_pooling = self.average_pooling(maps[:, :, :, tf.newaxis])
        else:
            average_pooling = maps

        flatten = self.flatten(average_pooling)
        context_feature = self.context_dense1(flatten)
        context_feature = tf.reshape(context_feature,
                                     [-1, self.args.obs_frames, 64])

        # concat, shape == (batch, obs, 128)
        concat_feature = self.concat([positions_embedding, context_feature])

        # transformer
        t_inputs = concat_feature
        t_outputs = positions

        # shape == (batch, obs, 128)
        t_features, _ = self.T1(t_inputs,
                                t_outputs,
                                training=training)

        # transfer GCN
        adj_matrix_transfer_T = self.adj_dense2(
            concat_feature)   # (batch, obs, pred)
        adj_matrix_transfer = tf.transpose(adj_matrix_transfer_T, [0, 2, 1])
        future_feature = GraphConv_func(t_features,
                                        adj_matrix_transfer,
                                        layer=self.gcn_transfer)

        # decoder
        predictions = self.decoder(future_feature)

        return predictions[:, :, tf.newaxis, :]


class MSNAlpha(BaseAgentStructure):
    MODEL_TYPE = MSNAlphaModel

    def __init__(self, terminal_args: list[str], manager=None):
        super().__init__(terminal_args, manager)

        # Force args
        self.args._set('key_points', str(self.args.pred_frames - 1))
        self.args._set('T', 'none')
