"""
@Author: Conghao Wong
@Date: 2022-09-13 21:18:29
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 18:57:46
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES, PROCESS_TYPES
from qpid.model import layers, transformer
from qpid.silverballers import (AgentArgs, BaseAgentModel, BaseAgentStructure,
                                BaseHandlerModel, BaseHandlerStructure,
                                HandlerArgs)


class MSNAlphaModel(BaseAgentModel):

    def __init__(self, Args: AgentArgs,
                 as_single_model: bool = True,
                 structure=None, *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        from qpid.mods.contextMaps.settings import (MAP_HALF_SIZE,
                                                    POOLING_BEFORE_SAVING)

        self.MAP_HALF_SIZE = MAP_HALF_SIZE
        self.POOLING_BEFORE_SAVING = POOLING_BEFORE_SAVING

        # Preprocess
        preprocess = []
        for index, operation in enumerate(["NONE",
                                           PROCESS_TYPES.SCALE,
                                           PROCESS_TYPES.ROTATE]):
            if self.args.preprocess[index] == '1':
                preprocess.append(operation)

        self.set_preprocess(*preprocess, **{PROCESS_TYPES.MOVE: 0})

        # Assign model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ, INPUT_TYPES.MAP)

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

        # Traj embedding, out shape == (batch, obs, 64)
        f_traj = self.pos_embedding(obs)

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
        concat_feature = torch.concat([f_traj, f_context], dim=-1)

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


class MSNAlpha(BaseAgentStructure):
    MODEL_TYPE = MSNAlphaModel

    def __init__(self, terminal_args: list[str], manager=None):
        super().__init__(terminal_args, manager)

        # Force args
        self.args._set('key_points', str(self.args.pred_frames - 1))
        self.args._set('T', 'none')


class MSNBetaModel(BaseHandlerModel):

    def __init__(self, Args: HandlerArgs,
                 as_single_model: bool = True,
                 structure=None, *args, **kwargs):

        from qpid.mods.contextMaps.settings import (MAP_HALF_SIZE,
                                                    POOLING_BEFORE_SAVING)

        self.MAP_HALF_SIZE = MAP_HALF_SIZE
        self.POOLING_BEFORE_SAVING = POOLING_BEFORE_SAVING

        # Force args
        Args._set('key_points', str(Args.pred_frames - 1))
        Args._set('T', 'none')

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        if self.args.model_type == 'frame-based':
            raise ValueError(self.args.model_type)

        # Preprocess
        preprocess = []
        for index, operation in enumerate(["NONE",
                                           PROCESS_TYPES.SCALE,
                                           PROCESS_TYPES.ROTATE]):
            if self.args.preprocess[index] == '1':
                preprocess.append(operation)

        self.set_preprocess(*preprocess, **{PROCESS_TYPES.MOVE: 0})

        # Assign model inputs
        # GT in the inputs is only used when training
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.MAP,
                        INPUT_TYPES.MAP_PARAS,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Layers
        # context feature
        if not self.POOLING_BEFORE_SAVING:
            self.average_pooling = layers.MaxPooling2D((5, 5))

        self.flatten = layers.Flatten(2)
        self.context_dense1 = layers.Dense(
            ((MAP_HALF_SIZE*2)//5)**2,
            (self.args.obs_frames+1) * 64,
            activation=torch.nn.Tanh)

        # Traj embedding
        self.pos_embedding = layers.Dense(2, 64, torch.nn.Tanh)

        # Linear interpolation
        self.linear_int = layers.interpolation.LinearPositionInterpolation()

        # Transformer is used as a backbone predictor
        self.transformer = transformer.Transformer(
            num_layers=4,
            d_model=128,
            num_heads=8,
            dff=512,
            input_vocab_size=2,
            target_vocab_size=2,
            pe_input=Args.obs_frames + 1,
            pe_target=Args.pred_frames
        )

    def forward(self, inputs: list[torch.Tensor],
                keypoints: torch.Tensor,
                keypoints_index: torch.Tensor,
                training=None, mask=None):

        # Unpack inputs
        obs = inputs[0]
        maps = inputs[1]
        dest = keypoints    # destinations

        # Concat positions and destinations
        positions = torch.concat([obs, dest], dim=-2)

        # Traj embedding, shape == (batch, obs+1, 64)
        f_traj = self.pos_embedding(positions)

        # Encoding context maps into context features
        # Shape of maps is (batch, 100, 100)
        # context feature, shape == (batch, obs+1, 64)
        if not self.POOLING_BEFORE_SAVING:
            average_pooling = self.average_pooling(maps[:, None])
        else:
            average_pooling = maps

        f_flatten = self.flatten(average_pooling)
        f_context = self.context_dense1(f_flatten)
        f_context = torch.reshape(f_context, [-1, self.args.obs_frames+1, 64])

        # Concat all features, shape == (batch, obs+1, 128)
        concat_feature = torch.concat([f_traj, f_context], dim=-1)

        # Linear interpolate future trajectories
        keypoints_index = torch.concat([torch.tensor([-1]), keypoints_index])
        keypoints = torch.concat([obs[..., -1:, :], keypoints], dim=-2)
        linear_traj = self.linear_int(keypoints_index, keypoints)

        # return t_outputs
        predictions, _ = self.transformer(concat_feature, linear_traj,
                                          training=training)

        return predictions


class MSNBeta(BaseHandlerStructure):
    MODEL_TYPE = MSNBetaModel
