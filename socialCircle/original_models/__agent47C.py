"""
@Author: Conghao Wong
@Date: 2022-06-20 21:40:38
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-06 20:40:14
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from qpid.model import layers, transformer
from qpid.silverballers import AgentArgs, BaseAgentModel, BaseAgentStructure


class Agent47CModel(BaseAgentModel):

    def __init__(self, Args: AgentArgs,
                 as_single_model: bool = True,
                 structure=None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Layers
        self.Tlayer, self.ITlayer = layers.get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = self.Tlayer(Oshape=(self.args.obs_frames, self.dim))
        self.it1 = self.ITlayer(Oshape=(self.n_key, self.dim))

        # Trajectory encoding
        self.te = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                      transform_layer=self.t1)

        # steps and shapes after applying transforms
        self.Tsteps_en = self.te.Tlayer.Tshape[0] if self.te.Tlayer else self.args.obs_frames
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Bilinear structure (outer product + pooling + fc)
        self.outer = layers.OuterLayer(self.d//2, self.d//2, reshape=False)
        self.pooling = layers.MaxPooling2D(pool_size=(2, 2),
                                           data_format='channels_first')
        self.flatten = layers.Flatten(axes_num=2)
        self.outer_fc = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Random id encoding
        self.ie = layers.TrajEncoding(self.d//2, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

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
        # It is used to generate multi-style predictions
        self.adj_fc = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.gcn = layers.GraphConv(units=self.d)

        # Decoder layers (with spectrums)
        self.decoder_fc1 = tf.keras.layers.Dense(self.d, tf.nn.tanh)
        self.decoder_fc2 = tf.keras.layers.Dense(
            self.Tsteps_de * self.Tchannels_de)

    def call(self, inputs: list[tf.Tensor],
             training=None, mask=None):
        """
        Run the first stage `agent47C` model.

        :param inputs: a list of tensors, including `trajs`
            - a batch of observed trajs, shape is `(..., obs, dim)`

        :param training: set to `True` when training, or leave it `None`

        :return predictions: predicted keypoints, \
            shape = `(..., Kc, N_key, dim)`
        """

        # unpack inputs
        trajs = inputs[0]   # (..., obs, dim)

        # feature embedding and encoding -> (..., Tsteps, d/2)
        # uses bilinear structure to encode features
        f = self.te(trajs)                  # (..., Tsteps, d/2)
        f = self.outer(f, f)                # (..., Tsteps, d/2, d/2)
        f = self.pooling(f)                 # (..., Tsteps, d/4, d/4)
        f = self.flatten(f)                 # (..., Tsteps, d*d/16)
        spec_features = self.outer_fc(f)    # (..., Tsteps, d/2)

        # Sample random predictions
        all_predictions = []
        rep_time = self.args.K_train if training else self.args.K

        t_outputs = self.t1(trajs)  # (..., Tsteps, Tchannels)

        for _ in range(rep_time):
            if not self.args.deterministic:
                # assign random ids and embedding -> (..., Tsteps, d)
                ids = tf.random.normal(
                    list(tf.shape(trajs)[:-2]) + [self.Tsteps_en, self.d_id])
                id_features = self.ie(ids)

                # transformer inputs
                # shapes are (..., Tsteps, d)
                t_inputs = self.concat([spec_features, id_features])

            else:
                t_inputs = spec_features

            # transformer -> (..., Tsteps, d)
            behavior_features, _ = self.T(inputs=t_inputs,
                                          targets=t_outputs,
                                          training=training)

            # multi-style features -> (..., Kc, d)
            adj = self.adj_fc(t_inputs)
            i = list(tf.range(adj.ndim))
            adj = tf.transpose(adj, i[:-2] + [i[-1], i[-2]])
            m_features = self.gcn(behavior_features, adj)

            # predicted keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            y = self.decoder_fc1(m_features)
            y = self.decoder_fc2(y)
            y = tf.reshape(y, list(tf.shape(y)[:-1]) +
                           [self.Tsteps_de, self.Tchannels_de])

            y = self.it1(y)
            all_predictions.append(y)

        return tf.concat(all_predictions, axis=-3)


class Agent47C(BaseAgentStructure):

    """
    Training structure for the `Agent47C` model.
    Note that it is only used to train the single model.
    Please use the `Silverballers` structure if you want to test any
    agent-handler based silverballers models.
    """

    def __init__(self, terminal_args: list[str], manager=None):
        super().__init__(terminal_args, manager)

        self.set_model_type(new_type=Agent47CModel)
