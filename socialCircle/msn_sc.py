"""
@Author: Conghao Wong
@Date: 2023-08-21 19:47:50
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 18:54:15
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES, PROCESS_TYPES
from qpid.model import layers, transformer
from qpid.silverballers import AgentArgs, BaseAgentStructure

from .__base import BaseSocialCircleModel, BaseSocialCircleStructure
from .__layers import SocialCircleLayer


class MSNSCModel(BaseSocialCircleModel):

    def __init__(self, Args: AgentArgs, as_single_model: bool = True,
                 structure: BaseAgentStructure = None, *args, **kwargs):

        from qpid.mods.contextMaps.settings import (MAP_HALF_SIZE,
                                                    POOLING_BEFORE_SAVING)

        self.MAP_HALF_SIZE = MAP_HALF_SIZE
        self.POOLING_BEFORE_SAVING = POOLING_BEFORE_SAVING

        # Force args
        Args._set('key_points', str(Args.pred_frames - 1))
        Args._set('T', 'none')

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Preprocess
        preprocess = []
        for index, operation in enumerate(["NONE",
                                           PROCESS_TYPES.SCALE,
                                           PROCESS_TYPES.ROTATE]):
            if self.args.preprocess[index] == '1':
                preprocess.append(operation)

        self.set_preprocess(*preprocess, **{PROCESS_TYPES.MOVE: 0})

        # Assign model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.MAP,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        # context feature
        if not POOLING_BEFORE_SAVING:
            self.average_pooling = layers.MaxPooling2D((5, 5))

        self.flatten = layers.Flatten(2)
        self.context_dense1 = layers.Dense(
            ((MAP_HALF_SIZE*2)//5)**2,
            self.args.obs_frames * 64,
            activation=torch.nn.Tanh)

        # traj embedding
        self.pos_embedding = layers.Dense(2, 64, torch.nn.Tanh)

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
        self.tse = layers.TrajEncoding(self.sc.dim, 64, torch.nn.ReLU,
                                       transform_layer=self.ts)

        # Concat and fuse SC
        self.concat_fc = layers.Dense(128, 64, torch.nn.Tanh)

        # Transformer is used as a feature extractor
        self.T1 = transformer.Transformer(
            num_layers=4,
            d_model=128,
            num_heads=8,
            dff=512,
            input_vocab_size=2,
            target_vocab_size=2,
            pe_input=Args.obs_frames,
            pe_target=Args.obs_frames,
            include_top=False
        )

        # Trainable adj matrix and gcn layer
        # It is used to generate multiple predictions within one model implementation
        self.ms_fc = layers.Dense(128, self.args.Kc, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(128, 128)

        # decoder
        self.decoder = layers.Dense(128, 2)

    def forward(self, inputs: list[torch.Tensor], training=None, *args, **kwargs):
        # Unpack inputs
        obs = inputs[0]
        maps = inputs[1]
        nei = inputs[2]

        # Start computing the SocialCircle
        # Compute and encode the SocialCircle
        social_circle, _ = self.sc(obs, nei)
        f_social = self.tse(social_circle)    # (batch, obs, 64)

        # Traj embedding, out shape == (batch, obs, 64)
        f_traj = self.pos_embedding(obs)

        # Feature fusion
        f_behavior = torch.concat([f_traj, f_social], dim=-1)
        f_behavior = self.concat_fc(f_behavior)

        # Encoding context maps into context features
        # Shape of maps is (batch, 100, 100)
        # context feature, shape == (batch, obs, 64)
        if not self.POOLING_BEFORE_SAVING:
            average_pooling = self.average_pooling(maps[:, None])
        else:
            average_pooling = maps

        f_flatten = self.flatten(average_pooling)
        f_context = self.context_dense1(f_flatten)
        f_context = torch.reshape(f_context, [-1, self.args.obs_frames, 64])

        # Concat all features, shape == (batch, obs, 128)
        concat_feature = torch.concat([f_behavior, f_context], dim=-1)

        # Transformer output shape is (batch, obs, 128)
        f_tran, _ = self.T1(inputs=concat_feature,
                            targets=obs,
                            training=training)

        # Multiple generations
        adj = self.ms_fc(concat_feature)        # (batch, obs, pred)
        adj = torch.transpose(adj, -1, -2)
        f_multi = self.ms_conv(f_tran, adj)     # (batch, Kc, 128)

        # Forecast destinations
        predictions = self.decoder(f_multi)         # (batch, Kc, 2)
        return predictions[..., None, :]


class MSNSCStructure(BaseSocialCircleStructure):
    MODEL_TYPE = MSNSCModel
