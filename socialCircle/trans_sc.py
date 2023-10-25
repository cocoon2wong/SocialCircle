"""
@Author: Conghao Wong
@Date: 2023-08-15 20:30:51
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 18:54:40
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES, PROCESS_TYPES
from qpid.model import layers, transformer
from qpid.silverballers import AgentArgs, BaseAgentStructure
from qpid.training import Structure

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

    def __init__(self, Args: AgentArgs, as_single_model: bool = True,
                 structure: BaseAgentStructure = None, *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Preprocess
        self.set_preprocess(**{PROCESS_TYPES.MOVE: 0})

        # Assign model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        tlayer, itlayer = layers.get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((self.args.pred_frames, self.dim))

        # Trajectory embedding
        if type(self.t1) == layers.transfroms.NoneTransformLayer:
            self.te = layers.TrajEncoding(self.dim, self.d//2,
                                          torch.nn.Tanh)
        else:
            self.te = layers.TrajEncoding(self.dim, self.d//2,
                                          torch.nn.Tanh, self.t1)

        # SocialCircle encoding
        tslayer, _ = layers.get_transform_layers(self.sc_args.Ts)
        self.sc = SocialCircleLayer(partitions=self.sc_args.partitions,
                                    max_partitions=self.args.obs_frames,
                                    use_velocity=self.sc_args.use_velocity,
                                    use_distance=self.sc_args.use_distance,
                                    use_direction=self.sc_args.use_direction,
                                    relative_velocity=self.sc_args.rel_speed,
                                    use_move_direction=self.sc_args.use_move_direction)
        self.ts = tslayer((self.args.obs_frames, self.sc.dim))
        self.tse = layers.TrajEncoding(self.sc.dim, self.d//2, torch.nn.ReLU,
                                       transform_layer=self.ts)

        # Concat and fuse SC
        self.concat_fc = layers.Dense(self.d, self.d//2, torch.nn.Tanh)

        # Steps and shapes after applying transforms
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Transformer is used as a feature extractor
        self.T = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.Tchannels_en,
            target_vocab_size=self.Tchannels_de,
            pe_input=self.Tsteps_en,
            pe_target=self.Tsteps_en,
            include_top=False
        )

        self.ms_fc = layers.Dense(self.d, self.Tsteps_de, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(self.d, self.d)

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_id, self.d//2, torch.nn.Tanh)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(self.d, 2*self.d, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(2*self.d, self.Tchannels_de)

    def forward(self, inputs: list[torch.Tensor], training=None, *args, **kwargs):
        # Unpack inputs
        obs = inputs[0]
        nei = inputs[1]

        # Start computing the SocialCircle
        # SocialCircle will be computed on each agent's center point
        c_obs = self.picker.get_center(obs)[..., :2]
        c_nei = self.picker.get_center(nei)[..., :2]

        # Compute and encode the SocialCircle
        social_circle, f_direction = self.sc(c_obs, c_nei)
        f_social = self.tse(social_circle)    # (batch, steps, d/2)

        # feature embedding and encoding -> (batch, obs, d)
        f_traj = self.te(obs)

        # Feature fusion
        f_behavior = torch.concat([f_traj, f_social], dim=-1)
        f_behavior = self.concat_fc(f_behavior)

        # Sampling random noise vectors
        all_predictions = []
        repeats = 1

        traj_targets = self.t1(obs)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f_behavior.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(obs.device))

            # Transformer inputs -> (batch, steps, d)
            f_final = torch.concat([f_behavior, f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Generations -> (batch, pred_steps, d)
            adj = self.ms_fc(f_final)
            adj = torch.transpose(adj, -1, -2)
            f_multi = self.ms_conv(f_tran, adj)     # (batch, pred_steps, d)

            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)

            y = self.it1(y)
            all_predictions.append(y)

        return torch.concat(all_predictions, dim=-3)   # (batch, 1, pred, dim)


class TransformerSCStructure(BaseSocialCircleStructure):
    MODEL_TYPE = TransformerSCModel

    def __init__(self, terminal_args, manager: Structure = None):
        super().__init__(terminal_args, manager)

        # Force args
        self.args._set('key_points', '_'.join(
            [str(i) for i in range(self.args.pred_frames)]))
