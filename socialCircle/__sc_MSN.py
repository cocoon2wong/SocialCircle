"""
@Author: Conghao Wong
@Date: 2023-08-21 19:47:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-06 20:25:05
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from qpid.constant import INPUT_TYPES, PROCESS_TYPES
from qpid.model import layers, transformer
from qpid.training import Structure
from qpid.utils import POOLING_BEFORE_SAVING

from .__args import SocialCircleArgs
from .__base import BaseSocialCircleModel, BaseSocialCircleStructure
from .__layers import SocialCircleLayer


def GraphConv_layer(output_units, activation=None):
    return tf.keras.layers.Dense(output_units, activation)


def GraphConv_func(features, A, output_units=64, activation=None, layer=None):
    dot = tf.matmul(A, features)
    if layer == None:
        res = tf.keras.layers.Dense(output_units, activation)(dot)
    else:
        res = layer(dot)
    return res


class MSNSCModel(BaseSocialCircleModel):

    def __init__(self, Args: SocialCircleArgs,
                 as_single_model: bool = True,
                 structure=None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Preprocess
        preprocess = []
        for index, operation in enumerate(["NONE",
                                           PROCESS_TYPES.SCALE,
                                           PROCESS_TYPES.ROTATE]):
            if self.args.preprocess[index] == '1':
                preprocess.append(operation)

        self.set_preprocess(*preprocess, **{PROCESS_TYPES.MOVE: 0})

        # Assign model inputs and preprocess layers
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.MAP,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        # context feature
        if not POOLING_BEFORE_SAVING:
            self.average_pooling = tf.keras.layers.AveragePooling2D(
                [5, 5], input_shape=[100, 100, 1])

        self.flatten = tf.keras.layers.Flatten()
        self.context_dense1 = tf.keras.layers.Dense(self.args.obs_frames * 64,
                                                    activation=tf.nn.tanh)

        # traj embedding
        self.pos_embedding = tf.keras.layers.Dense(64, tf.nn.tanh)

        # SocialCircle and fusion layers
        tslayer, _ = layers.get_transform_layers(self.args.Ts)
        self.ts = tslayer((self.args.obs_frames, 2))
        self.sc = SocialCircleLayer(partitions=self.args.partitions,
                                    max_partitions=self.args.obs_frames,
                                    use_velocity=self.args.use_velocity,
                                    use_distance=self.args.use_distance,
                                    use_direction=self.args.use_direction,
                                    relative_velocity=self.args.rel_speed,
                                    use_move_direction=self.args.use_move_direction)
        self.tse = layers.TrajEncoding(64, tf.nn.relu,
                                       transform_layer=self.ts)
        self.concat_fc = tf.keras.layers.Dense(64, tf.nn.tanh)

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
        nei = inputs[2]

        # traj embedding, shape == (batch, obs, 64)
        positions_embedding = self.pos_embedding(positions)

        # Compute and encode the SocialCircle
        social_circle, _ = self.sc(positions, nei)
        f_social = self.tse(social_circle)    # (batch, obs, 64)

        f_behavior = tf.concat([positions_embedding, f_social], axis=-1)
        f_behavior = self.concat_fc(f_behavior)

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
        concat_feature = self.concat([f_behavior, context_feature])

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


class MSNSCStructure(BaseSocialCircleStructure):
    MODEL_TYPE = MSNSCModel

    def __init__(self, terminal_args, manager: Structure = None):
        super().__init__(terminal_args, manager)

        # Force args
        self.args._set('key_points', str(self.args.pred_frames - 1))
        self.args._set('T', 'none')
