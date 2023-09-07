"""
@Author: Conghao Wong
@Date: 2023-08-15 20:30:51
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-06 20:54:54
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from qpid.constant import INPUT_TYPES, PROCESS_TYPES
from qpid.model import layers, transformer
from qpid.training import Structure

from .__args import SocialCircleArgs
from .__base import BaseSocialCircleModel, BaseSocialCircleStructure
from .__layers import SocialCircleLayer


class TransformerSCModel(BaseSocialCircleModel):
    """
    A simple Transformer-based trajectory prediction model.
    It takes the SocialCircle to model social interactions.

    - considers nothing about other interactions;
    - no keypoints-interpolation two-stage subnetworks;
    - contains only the backbone;
    - considers nothing about agents' multimodality.
    """

    def __init__(self, Args: SocialCircleArgs, as_single_model: bool = True,
                 structure=None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Preprocess
        self.set_preprocess(**{PROCESS_TYPES.MOVE: 0})

        # Assign model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        self.Tlayer, self.ITlayer = layers.get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = self.Tlayer(Oshape=(self.args.obs_frames, self.dim))
        self.it1 = self.ITlayer(Oshape=(self.args.pred_frames, self.dim))

        # Trajectory embedding
        if type(self.t1) == layers.transfroms.NoneTransformLayer:
            self.te = layers.TrajEncoding(
                self.d//2, tf.nn.relu, transform_layer=None)
        else:
            self.te = layers.TrajEncoding(self.d//2, tf.nn.relu, self.t1)

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

        self.Tsteps_en = self.t1.Tshape[0]
        self.Osteps_de = self.args.pred_frames
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        self.T = transformer.Transformer(num_layers=4,
                                         d_model=self.d,
                                         num_heads=8,
                                         dff=512,
                                         input_vocab_size=None,
                                         target_vocab_size=None,
                                         pe_input=self.Tsteps_en,
                                         pe_target=self.Tsteps_en,
                                         include_top=False)

        self.adj_fc = tf.keras.layers.Dense(self.Tsteps_de, tf.nn.tanh)
        self.gcn = layers.GraphConv(units=self.d)

        # Random id encoding
        self.ie = layers.TrajEncoding(self.d//2, tf.nn.tanh)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        # Decoder layers (with spectrums)
        self.decoder_fc1 = tf.keras.layers.Dense(2*self.d, tf.nn.tanh)
        self.decoder_fc2 = tf.keras.layers.Dense(self.Tchannels_de)

    def call(self, inputs, training=None, mask=None, *args, **kwargs):

        # unpack inputs
        trajs = inputs[0]
        nei = inputs[1]
        bs = trajs.shape[0]

        # Embed trajectories
        f = self.te(trajs)     # (batch, obs, d/2)

        # Compute and encode the SocialCircle
        social_circle, _ = self.sc(trajs, nei)
        f_social = self.tse(social_circle)    # (batch, obs, d/2)

        f_behavior = tf.concat([f, f_social], axis=-1)
        f_behavior = self.concat_fc(f_behavior)

        # Sample random predictions
        all_predictions = []
        rep_time = 1
        t_outputs = self.t1(trajs)

        for _ in range(rep_time):

            ids = tf.random.normal([bs, self.Tsteps_en, self.d_id])
            id_features = self.ie(ids)

            t_inputs = self.concat([f_behavior, id_features])
            t_features, _ = self.T(t_inputs, t_outputs, training)

            adj = tf.transpose(self.adj_fc(t_inputs), [0, 2, 1])
            t_features = self.gcn(t_features, adj)

            y = self.decoder_fc1(t_features)
            y = self.decoder_fc2(y)

            y = self.it1(y)

            all_predictions.append(y)

        return tf.stack(all_predictions, axis=1)


class TransformerSCStructure(BaseSocialCircleStructure):
    MODEL_TYPE = TransformerSCModel

    def __init__(self, terminal_args, manager: Structure = None):
        super().__init__(terminal_args, manager)

        # Force args
        self.args._set('key_points', '_'.join(
            [str(i) for i in range(self.args.pred_frames)]))
