"""
@Author: Conghao Wong
@Date: 2022-06-23 10:23:53
@LastEditors: Conghao Wong
@LastEditTime: 2023-08-16 17:19:23
@Description: Second stage V^2-Net model.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from qpid.constant import ANN_TYPES
from qpid.model import layers
from qpid.model.transformer import Transformer
from qpid.silverballers import (BaseHandlerModel, BaseHandlerStructure,
                                HandlerArgs)
from qpid.training import Structure


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
                 structure: Structure = None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

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
        self.te = layers.TrajEncoding(units=self.d//2,
                                      activation=tf.nn.tanh,
                                      transform_layer=self.t_layer)

        self.ce = layers.ContextEncoding(units=self.d//2,
                                         output_channels=input_Tsteps,
                                         activation=tf.nn.tanh)

        self.transformer = Transformer(num_layers=4,
                                       d_model=self.d,
                                       num_heads=8,
                                       dff=512,
                                       input_vocab_size=None,
                                       target_vocab_size=Tchannels,
                                       pe_input=input_Tsteps,
                                       pe_target=output_Tsteps,
                                       include_top=True)

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None,
             *args, **kwargs):

        # unpack inputs
        trajs_md, maps = inputs[:2]

        # Reshape keypoints to (..., K, steps, dim)
        if keypoints.ndim == trajs_md.ndim:
            keypoints = keypoints[..., tf.newaxis, :, :]

        trajs_md = tf.repeat(trajs_md[..., tf.newaxis, :, :],
                             repeats=tf.shape(keypoints)[-3],
                             axis=-3)

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
        t_inputs = tf.concat([traj_feature, context_feature], axis=-1)

        # transformer target shape = (batch, output_Tsteps, Tchannels)

        keypoints_index = tf.concat([[-1], keypoints_index], axis=0)
        keypoints = tf.concat([trajs[..., -1:, :], keypoints], axis=-2)

        # Add the last obs point to finish linear interpolation
        linear_pred = self.linear_int(keypoints_index, keypoints)

        traj = tf.concat([trajs, linear_pred], axis=-2)
        t_outputs = self.t_layer(traj)

        # transformer output shape = (batch, output_Tsteps, Tchannels)
        t_inputs = tf.repeat(t_inputs[..., tf.newaxis, :, :],
                             repeats=tf.shape(keypoints)[-3],
                             axis=-3)
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
        keypoints_md = tf.concat(
            [trajs_md[..., -1:, :], keypoints_md], axis=-2)
        l = self.linear_int(keypoints_index, keypoints_md)

        # Linear center points
        l_center = self.picker.get_center(l)[tf.newaxis]
        l_co = tf.cast(self.picker.get_coordinate_series(l), tf.float32)

        # Bias to the center points
        bias_center = l_co - l_center
        bias_linear = (y - linear_pred)[tf.newaxis]
        new_center = y[tf.newaxis]

        y_md = new_center + bias_center + bias_linear   # (M, batch, pred, 2)
        y_md = tf.concat(list(y_md), axis=-1)
        return y_md


class VB(BaseHandlerStructure):
    """
    Training structure for the second stage sub-network
    """
    MODEL_TYPE = VBModel
