"""
@Author: Conghao Wong
@Date: 2022-07-05 16:00:26
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-06 20:44:05
@Description: First stage V^2-Net model.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from qpid.model import layers, transformer
from qpid.silverballers import AgentArgs, BaseAgentModel, BaseAgentStructure


class VAModel(BaseAgentModel):
    """
    Keypoints Estimation Sub-network
    ---

    The first stage V^2-Net sub-network.
    It is used to model agents' global plannings by considering
    agents' observed trajectory spectrums.
    The model takes agents' observed trajectories as the input,
    and output several keypoint trajectory spectrums finally.
    FFTs are applied before and after the model implementing.
    """

    def __init__(self, Args: AgentArgs,
                 as_single_model: bool = True,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Layers
        self.Tlayer, self.ITlayer = layers.get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = self.Tlayer((self.args.obs_frames, self.dim))
        self.it1 = self.ITlayer((self.n_key, self.dim))

        # Trajectory encoding
        self.te = layers.TrajEncoding(self.d//2,
                                      tf.nn.tanh,
                                      transform_layer=self.t1)

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
        bs = trajs.shape[0]

        # feature embedding and encoding -> (batch, obs, d)
        spec_features = self.te(trajs)

        all_predictions = []
        rep_time = self.args.K_train if training else self.args.K
        for _ in range(rep_time):
            # assign random ids and embedding -> (batch, obs, d)
            ids = tf.random.normal([bs, self.Tsteps_en, self.d_id])
            id_features = self.ie(ids)

            # transformer inputs
            t_inputs = self.concat([spec_features, id_features])
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


class VA(BaseAgentStructure):
    """
    Training structure for the first stage sub-network
    """

    def __init__(self, terminal_args: list[str], manager=None):
        super().__init__(terminal_args, manager)
        self.set_model_type(new_type=VAModel)
