"""
@Author: Conghao Wong
@Date: 2023-08-08 15:26:35
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-06 20:54:44
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


class EVSCModel(BaseSocialCircleModel):

    def __init__(self, Args: SocialCircleArgs, as_single_model: bool = True,
                 structure: BaseSocialCircleStructure = None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Assign model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        tlayer, itlayer = layers.get_transform_layers(self.args.T)
        tslayer, _ = layers.get_transform_layers(self.args.Ts)

        # Transform layers
        self.ts = tslayer((self.args.obs_frames, 2))
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((self.n_key, self.dim))

        # Trajectory encoding
        self.te = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                      transform_layer=self.t1)

        # SocialCircle encoding
        self.sc = SocialCircleLayer(partitions=self.args.partitions,
                                    max_partitions=self.args.obs_frames,
                                    use_velocity=self.args.use_velocity,
                                    use_distance=self.args.use_distance,
                                    use_direction=self.args.use_direction,
                                    relative_velocity=self.args.rel_speed,
                                    use_move_direction=self.args.use_move_direction)
        self.tse = layers.TrajEncoding(self.d//2, tf.nn.relu,
                                       transform_layer=self.ts)

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Bilinear structure (outer product + pooling + fc)
        # For trajectories
        self.outer = layers.OuterLayer(self.d//2, self.d//2)
        self.pooling = layers.MaxPooling2D(
            (2, 2), data_format='channels_first')
        self.flatten = layers.Flatten(axes_num=2)
        self.outer_fc = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d//2, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.concat_fc = tf.keras.layers.Dense(self.d//2, tf.nn.tanh)

        # Transformer backbone
        self.T = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=None,
            target_vocab_size=None,
            pe_input=self.Tsteps_en,
            pe_target=self.Tsteps_en,
            include_top=False
        )

        # Multi-style prediction
        self.ms_fc = tf.keras.layers.Dense(self.args.Kc, tf.nn.tanh)
        self.ms_conv = layers.GraphConv(self.d)

        # Decoder layers
        self.decoder_fc1 = tf.keras.layers.Dense(self.d, tf.nn.tanh)
        self.decoder_fc2 = tf.keras.layers.Dense(
            self.Tsteps_de * self.Tchannels_de
        )

    def call(self, inputs, training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        obs = inputs[0]     # (batch, obs, dim)
        nei = inputs[1]     # (batch, a:=max_agents, obs, dim)

        # Start computing the SocialCircle
        # SocialCircle will be computed on each agent's center point
        c_obs = self.picker.get_center(obs)[..., :2]
        c_nei = self.picker.get_center(nei)[..., :2]

        # Compute and encode the SocialCircle
        social_circle, f_direction = self.sc(c_obs, c_nei)
        f_social = self.tse(social_circle)    # (batch, steps, d/2)

        # Trajectory embedding and encoding
        f = self.te(obs)
        f = self.outer(f, f)
        f = self.pooling(f)
        f = self.flatten(f)
        f_traj = self.outer_fc(f)       # (batch, steps, d/2)

        # Feature fusion
        f_behavior = tf.concat([f_traj, f_social], axis=-1)
        f_behavior = self.concat_fc(f_behavior)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K
        traj_targets = self.t1(obs)

        for _ in range(repeats):
            z = tf.random.normal(list(tf.shape(f_behavior)[:-1]) + [self.d_id])
            f_z = self.ie(z)
            # (batch, steps, 2*d)
            f_final = tf.concat([f_behavior, f_z], axis=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Multiple generations
            adj = self.ms_fc(f_final)      # (batch, steps, Kc)
            i = list(tf.range(adj.ndim))
            adj = tf.transpose(adj, i[:-2] + [i[-1], i[-2]])
            f_multi = self.ms_conv(f_tran, adj)    # (b, Kc, d)

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)
            y = tf.reshape(y, list(tf.shape(y)[:-1]) +
                           [self.Tsteps_de, self.Tchannels_de])

            y = self.it1(y)
            all_predictions.append(y)

        Y = tf.concat(all_predictions, axis=-3)   # (batch, K, n_key, dim)
        return Y, social_circle, f_direction


class EVSCStructure(BaseSocialCircleStructure):
    MODEL_TYPE = EVSCModel
