"""
@Author: Conghao Wong
@Date: 2022-07-05 16:00:26
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 18:59:01
@Description: First stage V^2-Net model.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import ANN_TYPES
from qpid.model import layers, transformer
from qpid.model.transformer import Transformer
from qpid.silverballers import (AgentArgs, BaseAgentModel, BaseAgentStructure,
                                BaseHandlerModel, BaseHandlerStructure,
                                HandlerArgs)


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
                 structure=None, *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # Layers
        tlayer, itlayer = layers.get_transform_layers(self.args.T)

        # Transform layers
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((self.n_key, self.dim))

        # Trajectory encoding
        self.te = layers.TrajEncoding(self.dim, self.d//2,
                                      torch.nn.Tanh,
                                      transform_layer=self.t1)

        # steps and shapes after applying transforms
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_id, self.d//2, torch.nn.Tanh)

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

        # Trainable adj matrix and gcn layer
        # See our previous work "MSN: Multi-Style Network for Trajectory Prediction" for detail
        # It is used to generate multiple predictions within one model implementation
        self.ms_fc = layers.Dense(self.d, self.args.Kc, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(self.d, self.d)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(self.d, self.d, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(self.d,
                                        self.Tsteps_de * self.Tchannels_de)

    def forward(self, inputs: list[torch.Tensor], training=None, *args, **kwargs):
        # Unpack inputs
        obs = inputs[0]     # (batch, obs, dim)

        # Feature embedding and encoding -> (batch, obs, d/2)
        f_traj = self.te(obs)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        traj_targets = self.t1(obs)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f_traj.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(obs.device))

            # Transformer inputs -> (batch, steps, d)
            f_final = torch.concat([f_traj, f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Multiple generations -> (batch, Kc, d)
            adj = self.ms_fc(f_final)               # (batch, steps, Kc)
            adj = torch.transpose(adj, -1, -2)
            f_multi = self.ms_conv(f_tran, adj)     # (batch, Kc, d)

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)
            y = torch.reshape(y, list(y.shape[:-1]) +
                              [self.Tsteps_de, self.Tchannels_de])

            y = self.it1(y)
            all_predictions.append(y)

        return torch.concat(all_predictions, dim=-3)   # (batch, K, n_key, dim)


class VBModel(BaseHandlerModel):
    """
    Spectrum Interpolation Sub-network
    ---

    The second stage V^2-Net sub-network.
    It is used to interpolate agents' entire predictions
    by considering their interactions details.
    It also implements on agents' spectrums instead of
    their trajectories.
    """

    def __init__(self, Args: HandlerArgs,
                 as_single_model: bool = True,
                 structure=None, *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        from qpid.mods.contextMaps import ContextEncoding

        self.accept_batchK_inputs = True

        if self.args.model_type == 'frame-based':
            raise ValueError(self.args.model_type)

        # Transform layers
        input_steps = self.args.obs_frames
        output_steps = self.args.obs_frames + self.args.pred_frames

        Tlayer, ITlayer = layers.get_transform_layers(self.args.T)
        self.t_layer = Tlayer((input_steps, self.dim))
        self.it_layer = ITlayer((output_steps, 2))

        # Shapes
        input_Tsteps, Tchannels = self.t_layer.Tshape
        output_Tsteps, _ = self.it_layer.Tshape

        # Linear layer
        self.linear_int = layers.interpolation.LinearPositionInterpolation()

        # Encoding layers
        # NOTE: All the following layers are calculated
        #       in the ***frequency domain***.
        self.te = layers.TrajEncoding(self.dim, self.d//2,
                                      torch.nn.Tanh,
                                      transform_layer=self.t_layer)

        self.ce = ContextEncoding(units=self.d//2,
                                  output_channels=input_Tsteps,
                                  activation=torch.nn.Tanh)

        self.transformer = Transformer(num_layers=4,
                                       d_model=self.d,
                                       num_heads=8,
                                       dff=512,
                                       input_vocab_size=Tchannels,
                                       target_vocab_size=Tchannels,
                                       pe_input=input_Tsteps,
                                       pe_target=output_Tsteps,
                                       include_top=True)

    def forward(self, inputs: list[torch.Tensor],
                keypoints: torch.Tensor,
                keypoints_index: torch.Tensor,
                training=None, mask=None,
                *args, **kwargs):

        # unpack inputs
        trajs_md, maps = inputs[:2]

        # Reshape keypoints to (..., K, steps, dim)
        if keypoints.ndim == trajs_md.ndim:
            keypoints = keypoints[..., None, :, :]

        trajs_md = torch.repeat_interleave(trajs_md[..., None, :, :],
                                           repeats=keypoints.shape[-3],
                                           dim=-3)

        keypoints_md = keypoints

        # Only accept 2-dimensional trajectories
        trajs = self.picker.get_center(trajs_md)[..., :2]
        keypoints = self.picker.get_center(keypoints_md)[..., :2]

        # Embedding and encoding
        # Transformations are applied in `self.te`
        # (batch, input_Tsteps, d//2)
        traj_feature = self.te(trajs[..., 0, :, :])
        context_feature = self.ce(maps)  # (batch, input_Tsteps, d//2)

        # transformer inputs shape = (batch, input_Tsteps, d)
        t_inputs = torch.concat([traj_feature, context_feature], dim=-1)

        # transformer target shape = (batch, output_Tsteps, Tchannels)
        keypoints_index = torch.concat([torch.tensor([-1]), keypoints_index])
        keypoints = torch.concat([trajs[..., -1:, :], keypoints], dim=-2)

        # Add the last obs point to finish linear interpolation
        linear_pred = self.linear_int(keypoints_index, keypoints)

        traj = torch.concat([trajs, linear_pred], dim=-2)
        t_outputs = self.t_layer(traj)

        # transformer output shape = (batch, output_Tsteps, Tchannels)
        t_inputs = torch.repeat_interleave(t_inputs[..., None, :, :],
                                           repeats=keypoints.shape[-3],
                                           dim=-3)
        p_fft, _ = self.transformer(t_inputs,
                                    t_outputs,
                                    training=training)

        # Inverse transform
        p = self.it_layer(p_fft)
        y = p[..., self.args.obs_frames:, :]

        if training:
            if self.args.anntype != ANN_TYPES.CO_2D:
                self.log('This model only support 2D coordinate trajectories' +
                         ' when training. Annotation type received is' +
                         f'`{self.args.anntype}`,',
                         level='error', raiseError=ValueError)
            return y

        # Calculate linear prediction (M-dimensional)
        keypoints_md = torch.concat([trajs_md[..., -1:, :],
                                     keypoints_md], dim=-2)
        l: torch.Tensor = self.linear_int(keypoints_index, keypoints_md)

        # Linear center points
        l_center = self.picker.get_center(l)[None]
        l_co = torch.stack(self.picker.get_coordinate_series(l))

        # Bias to the center points
        bias_center = l_co - l_center
        bias_linear = (y - linear_pred)[None]
        new_center = y[None]

        y_md = new_center + bias_center + bias_linear   # (M, batch, pred, 2)
        y_md = torch.concat(list(y_md), dim=-1)
        return y_md


class VA(BaseAgentStructure):
    """
    Training structure for the first stage sub-network
    """
    MODEL_TYPE = VAModel


class VB(BaseHandlerStructure):
    """
    Training structure for the second stage sub-network
    """
    MODEL_TYPE = VBModel
