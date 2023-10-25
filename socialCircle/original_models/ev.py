"""
@Author: Conghao Wong
@Date: 2022-06-20 21:40:38
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-20 09:38:44
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers, transformer
from qpid.silverballers import AgentArgs, BaseAgentModel, BaseAgentStructure


class Agent47CModel(BaseAgentModel):

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
                                      torch.nn.ReLU,
                                      transform_layer=self.t1)

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Bilinear structure (outer product + pooling + fc)
        # For trajectories
        self.outer = layers.OuterLayer(self.d//2, self.d//2)
        self.pooling = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten(axes_num=2)
        self.outer_fc = layers.Dense((self.d//4)**2, self.d//2, torch.nn.Tanh)

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
        """
        Run the first stage `agent47C` model.

        :param inputs: a list of tensors, including `trajs`
            - a batch of observed trajs, shape is `(..., obs, dim)`

        :param training: set to `True` when training, or leave it `None`

        :return predictions: predicted keypoints, \
            shape = `(..., Kc, N_key, dim)`
        """

        # Unpack inputs
        obs = inputs[0]     # (batch, obs, dim)

        # Trajectory embedding and encoding
        f = self.te(obs)
        f = self.outer(f, f)
        f = self.pooling(f)
        f = self.flatten(f)
        f_traj = self.outer_fc(f)       # (batch, steps, d/2)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        traj_targets = self.t1(obs)

        for _ in range(repeats):
            if not self.args.deterministic:
                # Assign random ids and embedding -> (batch, steps, d)
                z = torch.normal(mean=0, std=1,
                                 size=list(f_traj.shape[:-1]) + [self.d_id])
                f_z = self.ie(z.to(obs.device))

                # (batch, steps, 2*d)
                f_final = torch.concat([f_traj, f_z], dim=-1)

            else:
                f_final = f_traj

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


class Agent47C(BaseAgentStructure):
    """
    Training structure for the `Agent47C` model.
    Note that it is only used to train the single model.
    Please use the `Silverballers` structure if you want to test any
    agent-handler based silverballers models.
    """
    MODEL_TYPE = Agent47CModel
