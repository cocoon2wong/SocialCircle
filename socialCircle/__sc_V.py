"""
@Author: Conghao Wong
@Date: 2023-08-15 19:08:05
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-06 20:54:22
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from qpid.constant import INPUT_TYPES
from qpid.model import layers, transformer

from .__args import SocialCircleArgs
from .__base import BaseSocialCircleModel, BaseSocialCircleStructure
from .__layers import SocialCircleLayer


class VSCModel(BaseSocialCircleModel):

    def __init__(self, Args: SocialCircleArgs, as_single_model: bool = True,
                 structure=None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Assign model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        self.Tlayer, self.ITlayer = layers.get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = self.Tlayer((self.args.obs_frames, self.dim))
        self.it1 = self.ITlayer((self.n_key, self.dim))

        # Trajectory encoding
        self.te = layers.TrajEncoding(self.d//2,
                                      tf.nn.tanh,
                                      transform_layer=self.t1)

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
        self.tse = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                       transform_layer=self.ts)
        self.concat_fc = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Noise encoding layers
        self.ie = layers.TrajEncoding(self.d//2,
                                      activation=tf.nn.tanh)

        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # steps and shapes after applying transforms
        self.Tsteps_en = self.t1.Tshape[0]
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Transformer is used as a feature extractor
        self.T = transformer.Transformer(num_layers=4,
                                         d_model=self.d,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=None,
                                         pe_input=self.Tsteps_en,
                                         pe_target=self.Tsteps_en,
                                         include_top=False)

        # Trainable adj matrix and gcn layer
        # See our previous work "MSN: Multi-Style Network for Trajectory Prediction" for detail
        # It is used to generate multiple predictions within one model implementation
        self.adj_fc = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.gcn = layers.GraphConv(units=self.d)

        # Decoder layers
        self.decoder_fc1 = tf.keras.layers.Dense(self.d, tf.nn.tanh)
        self.decoder_fc2 = tf.keras.layers.Dense(
            self.Tsteps_de * self.Tchannels_de)
        self.decoder_reshape = tf.keras.layers.Reshape(
            [self.args.Kc, self.Tsteps_de, self.Tchannels_de])

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None):

        # unpack inputs
        trajs = inputs[0]   # (batch, obs, 2)
        nei = inputs[1]
        bs = trajs.shape[0]

        # feature embedding and encoding -> (batch, obs, d)
        spec_features = self.te(trajs)

        # Compute and encode the SocialCircle
        c_obs = self.picker.get_center(trajs)[..., :2]
        c_nei = self.picker.get_center(nei)[..., :2]
        social_circle, _ = self.sc(c_obs, c_nei)
        f_social = self.tse(social_circle)    # (batch, steps, d/2)

        f_behavior = tf.concat([spec_features, f_social], axis=-1)
        f_behavior = self.concat_fc(f_behavior)

        all_predictions = []
        rep_time = self.args.K_train if training else self.args.K
        for _ in range(rep_time):
            # assign random ids and embedding -> (batch, obs, d)
            ids = tf.random.normal([bs, self.Tsteps_en, self.d_id])
            id_features = self.ie(ids)

            # transformer inputs
            t_inputs = self.concat([f_behavior, id_features])
            t_outputs = self.t1(trajs)

            # transformer -> (batch, obs, d)
            behavior_features, _ = self.T(inputs=t_inputs,
                                          targets=t_outputs,
                                          training=training)

            # features -> (batch, Kc, d)
            adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
            m_features = self.gcn(behavior_features, adj)

            # predicted keypoints -> (batch, Kc, key, 2)
            y = self.decoder_fc1(m_features)
            y = self.decoder_fc2(y)
            y = self.decoder_reshape(y)

            y = self.it1(y)
            all_predictions.append(y)

        return tf.concat(all_predictions, axis=1)


class VSCStructure(BaseSocialCircleStructure):
    MODEL_TYPE = VSCModel
