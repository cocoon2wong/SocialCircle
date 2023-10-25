"""
@Author: Conghao Wong
@Date: 2022-06-20 15:28:14
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-10 18:08:23
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import sys

import torch

import qpid
import socialCircle
from scripts.utils import get_value

torch.autograd.set_detect_anomaly(True)


def main(args: list[str], run_train_or_test=True):
    h_value = None

    if '--help' in args:
        h_value = get_value('--help', args, default='all_args')
    elif '-h' in args:
        h_value = get_value('-h', args, default='all_args')

    if h_value:
        qpid.print_help_info(h_value)
        exit()

    min_args = qpid.args.Args(terminal_args=args,
                              is_temporary=True)

    model = min_args.model
    if model == 'linear':
        s = qpid.applications.Linear
    else:
        s = qpid.silverballers.SILVERBALLERS_DICT.get_structure(model)

    t = s(terminal_args=args)

    if run_train_or_test:
        t.train_or_test()

    # It is used to debug
    # t.print_info_all()

    return t

if __name__ == '__main__':
    main(sys.argv)
