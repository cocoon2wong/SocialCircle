"""
@Author: Conghao Wong
@Date: 2023-09-06 20:45:28
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-12 15:44:02
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as __qpid

from .ev import Agent47C, Agent47CModel
from .msn import MSNAlpha, MSNAlphaModel, MSNBeta, MSNBetaModel
from .trans import MinimalV, MinimalVModel
from .v import VA, VB, VAModel, VBModel

__qpid.silverballers.register(
    # MSN
    msna=[MSNAlpha, MSNAlphaModel],
    msnb=[MSNBeta, MSNBetaModel],

    # V^2-Net
    va=[VA, VAModel],
    agent=[VA, VAModel],
    vb=[VB, VBModel],

    # E-V^2-Net
    eva=[Agent47C, Agent47CModel],

    # agent47 series
    agent47C=[Agent47C, Agent47CModel],

    # Other models
    trans=[MinimalV, MinimalVModel],
    mv=[MinimalV, MinimalVModel],
)
