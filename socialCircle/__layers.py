"""
@Author: Conghao Wong
@Date: 2023-08-08 14:55:56
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 18:48:22
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.utils import get_mask


class SocialCircleLayer(torch.nn.Module):

    def __init__(self, partitions: int,
                 max_partitions: int = None,
                 use_velocity=True,
                 use_distance=True,
                 use_direction=True,
                 use_move_direction=False,
                 mu=0.0001,
                 relative_velocity=False,
                 *args, **kwargs):
        """
        A layer to compute the SocialCircle Meta-components.

        ## Partition Settings
        :param partitions: The number of partitions in the circle.
        :param max_partitions: The number of partitions (after zero padding).

        ## SocialCircle Factors
        :param use_velocity: Choose whether to use the velocity factor.
        :param use_distance: Choose whether to use the distance factor.
        :param use_direction: Choose whether to use the direction factor.
        :param use_move_direction: Choose whether to use the move direction factor.

        ## SocialCircle Options
        :param relative_velocity: Choose whether to use relative velocity or not.
        :param mu: The small number to prevent dividing zero when computing. \
            It only works when `relative_velocity` is set to `True`.
        """
        super().__init__(*args, **kwargs)

        self.partitions = partitions
        self.max_partitions = max_partitions

        self.use_velocity = use_velocity
        self.use_distance = use_distance
        self.use_direction = use_direction

        self.rel_velocity = relative_velocity
        self.use_move_direction = use_move_direction
        self.mu = mu

    @property
    def dim(self) -> int:
        """
        The number of SocialCircle factors.
        """
        return int(self.use_velocity) + int(self.use_distance) + \
            int(self.use_direction) + int(self.use_move_direction)

    def forward(self, trajs, nei_trajs, *args, **kwargs):
        # Move vectors -> (batch, ..., 2)
        # `nei_trajs` are relative values to target agents' last obs step
        obs_vector = trajs[..., -1:, :] - trajs[..., 0:1, :]
        nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]
        nei_posion_vector = nei_trajs[..., -1, :]

        # Velocity factor
        if self.use_velocity:
            # Calculate velocities
            nei_velocity = torch.norm(nei_vector, dim=-1)    # (batch, n)
            obs_velocity = torch.norm(obs_vector, dim=-1)    # (batch, 1)

            # Speed factor in the SocialCircle
            if self.rel_velocity:
                f_velocity = (nei_velocity + self.mu)/(obs_velocity + self.mu)
            else:
                f_velocity = nei_velocity

        # Distance factor
        if self.use_distance:
            f_distance = torch.norm(nei_posion_vector, dim=-1)

        # Move direction factor
        if self.use_move_direction:
            obs_move_direction = torch.atan2(obs_vector[..., 0],
                                             obs_vector[..., 1])
            nei_move_direction = torch.atan2(nei_vector[..., 0],
                                             nei_vector[..., 1])
            delta_move_direction = nei_move_direction - obs_move_direction
            f_move_direction = delta_move_direction % (2*np.pi)

        # Direction factor
        f_direction = torch.atan2(nei_posion_vector[..., 0],
                                  nei_posion_vector[..., 1])
        f_direction = f_direction % (2*np.pi)

        # Angles (the independent variable \theta)
        angle_indices = f_direction / (2*np.pi/self.partitions)
        angle_indices = angle_indices.to(torch.int32)

        # Mask neighbors
        nei_mask = get_mask(torch.sum(nei_trajs, dim=[-1, -2]), torch.int32)
        angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

        # Compute the SocialCircle
        social_circle = []
        for ang in range(self.partitions):
            _mask = (angle_indices == ang).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1)

            n = _mask_count + 0.0001
            social_circle.append([])

            if self.use_velocity:
                _velocity = torch.sum(f_velocity * _mask, dim=-1) / n
                social_circle[-1].append(_velocity)

            if self.use_distance:
                _distance = torch.sum(f_distance * _mask, dim=-1) / n
                social_circle[-1].append(_distance)

            if self.use_direction:
                _direction = torch.sum(f_direction * _mask, dim=-1) / n
                social_circle[-1].append(_direction)

            if self.use_move_direction:
                _move_d = torch.sum(f_move_direction * _mask, dim=-1) / n
                social_circle[-1].append(_move_d)

        # Shape of the final SocialCircle: (batch, p, 3)
        social_circle = [torch.stack(i) for i in social_circle]
        social_circle = torch.stack(social_circle)
        social_circle = torch.permute(social_circle, [2, 0, 1])

        if (((m := self.max_partitions) is not None) and
                (m > (n := self.partitions))):
            paddings = [0, 0, 0, m - n, 0, 0]
            social_circle = torch.nn.functional.pad(social_circle, paddings)

        return social_circle, f_direction
