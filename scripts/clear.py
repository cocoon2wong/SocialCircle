"""
@Author: Conghao Wong
@Date: 2021-07-19 11:11:10
@LastEditors: Conghao Wong
@LastEditTime: 2023-10-11 11:00:39
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import sys

import numpy as np
from utils import get_value


def clean_logs(base_dir):
    """
    Delete all saved model weights except the best one.
    """
    for d in os.listdir(base_dir):
        cd = os.path.join(base_dir, d)

        if d.startswith('.') or not os.path.isdir(cd):
            continue

        files = os.listdir(cd)

        if (fn := 'best_ade_epoch.txt') in files:
            best_epoch = np.loadtxt(os.path.join(cd, fn))[1].astype(int)
            patterns = [f'_epoch{best_epoch}.tf',
                        f'_epoch{best_epoch}.pt']

        else:
            continue

        for f in files:
            path = os.path.join(cd, f)
            best_find = False

            for pattern in patterns:
                if pattern in f:
                    print(f'Find {path}.')
                    best_find = True
                    break

            if not best_find and (f.endswith('.tf.index')
                                  or ('.tf.data' in f)
                                  or f.endswith('.pt')):
                print(f'Remove {path}.')
                os.remove(path)


def clean_figs(base_dir):
    """
    Delete all saved visualizations in the `base_dir`.
    """
    for d in os.listdir(base_dir):
        cd = os.path.join(base_dir, d)

        if d.startswith('.') or not os.path.isdir(cd):
            continue

        files = os.listdir(cd)

        if (fn := 'VisualTrajs') in files:
            p = os.path.join(cd, fn)
            os.system(f'rm -r {p}')
            print(f'Removed `{p}`.')


def clean_dirs(base_dir):
    """
    Delete all empty dirs in the `base_dir`.
    """
    for d in os.listdir(base_dir):
        cd = os.path.join(base_dir, d)

        if d.startswith('.') or not os.path.isdir(cd):
            continue

        files = os.listdir(cd)

        if ((len(files) == 0) or
                (len(files) == 1 and files[0] == '.DS_Store')):
            os.system(f'rm -r {cd}')
            print(f'Removed `{cd}.`')


if __name__ == '__main__':
    args = sys.argv
    if '--logs' in args:
        clean_logs(get_value('--logs', args))

    elif '--figs' in args:
        clean_figs(get_value('--figs', args))

    elif '--dirs' in args:
        clean_dirs(get_value('--dirs', args))
