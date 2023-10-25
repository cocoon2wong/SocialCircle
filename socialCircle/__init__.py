"""
@Author: Conghao Wong
@Date: 2023-08-08 15:52:46
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-23 18:17:46
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as __qpid

from . import original_models
from .__args import SocialCircleArgs
from .__layers import SocialCircleLayer
from .ev_sc import EVSCModel, EVSCStructure
from .msn_sc import MSNSCModel, MSNSCStructure
from .trans_sc import TransformerSCModel, TransformerSCStructure
from .v_sc import VSCModel, VSCStructure

# Add new args
__qpid.register_new_args(SocialCircleArgs, 'SocialCircle Args', __package__)
__qpid.args.add_arg_alias('--sc', ['--model', 'MKII', '-lb', 'speed', '-la'])

# Register new models
__qpid.silverballers.register(evsc=[EVSCStructure, EVSCModel],
                              vsc=[VSCStructure, VSCModel],
                              msnsc=[MSNSCStructure, MSNSCModel],
                              transsc=[TransformerSCStructure,
                                       TransformerSCModel])

__qpid._log_mod_loaded(__package__)
