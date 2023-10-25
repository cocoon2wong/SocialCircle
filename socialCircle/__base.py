"""
@Author: Conghao Wong
@Date: 2023-08-08 15:57:43
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-17 18:56:09
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from qpid.silverballers import AgentArgs, BaseAgentModel, BaseAgentStructure
from qpid.training import Structure

from .__args import SocialCircleArgs


class BaseSocialCircleModel(BaseAgentModel):
    def __init__(self, Args: AgentArgs, as_single_model: bool = True,
                 structure: BaseAgentStructure = None, *args, **kwargs):
        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        self.sc_args = self.args._register_mod_args(SocialCircleArgs,
                                                    __package__)

    def print_info(self, **kwargs):
        factors = [item for item in ['velocity',
                                     'distance',
                                     'direction',
                                     'move_direction']
                   if getattr(self.sc_args, f'use_{item}')]

        info = {
            # 'Transform type (SocialCircle)': self.sc_args.Ts,
            'Partitions in SocialCircle': self.sc_args.partitions,
            'Max partitions in SocialCircle': self.args.obs_frames,
            'Factors used in SocialCircle': factors}

        kwargs.update(**info)
        return super().print_info(**kwargs)


class BaseSocialCircleStructure(BaseAgentStructure):

    def __init__(self, terminal_args: list[str] | AgentArgs,
                 manager: Structure = None):
        super().__init__(terminal_args, manager)

        if self.args.model_type != 'agent-based':
            self.log('SocialCircle models only support model type `agent-based`.' +
                     f' Current setting is `{self.args.model_type}`.',
                     level='error', raiseError=ValueError)
