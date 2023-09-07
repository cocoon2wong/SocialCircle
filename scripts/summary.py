"""
@Author: Conghao Wong
@Date: 2022-05-03 09:07:21
@LastEditors: Conghao Wong
@LastEditTime: 2022-11-11 09:35:59
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import json
import os
import sys

import numpy as np
from utils import get_value


def read_model_weights(base_path: str):

    model_dict = []

    for d in os.listdir(base_path):
        if d.startswith('.'):
            continue

        try:
            current_path = os.path.join(base_path, d)

            with open(os.path.join(current_path, 'args.json'), 'r') as f:
                args = json.load(f)

            metric = np.loadtxt(os.path.join(
                current_path, 'best_ade_epoch.txt'))[0]

        except:
            print(f'Skip path `{current_path}`')
            continue

        model_type = args['model']
        split = args['split']

        model_dict.append({'model': model_type,
                           'split': split,
                           'metric': metric,
                           'path': current_path})

    return model_dict


def sort_weights(base_path: str, model_dict: list[dict], sort_key: str = 'model'):

    sorted_dict: dict[str, list[int]] = {}
    for index, item in enumerate(model_dict):
        sort_value = item[sort_key]
        ds = item['split']

        if not sort_value in sorted_dict.keys():
            sorted_dict[sort_value] = []

        sorted_dict[sort_value].append(index)

    summary_dict = {}
    for ds in sorted(sorted_dict.keys()):
        items = [model_dict[i] for i in sorted_dict[ds]]
        items = sorted(items, key=lambda x: x['metric'])
        summary_dict[ds] = items

    p = os.path.join(base_path, 'summary.json')

    if not os.path.exists(p_temp := './summary'):
        os.mkdir(p_temp)

    p_temp = os.path.join(p_temp, 'summary_{}.json'.format(
        base_path.replace('/', '_')))

    for path in [p, p_temp]:
        with open(path, 'w+') as f:
            json.dump(summary_dict, f, indent=4)


if __name__ == '__main__':
    args = sys.argv
    if '--logs' in args:
        base_path = get_value('--logs', args)
    else:
        base_path = './logs'

    if '--sort' in args:
        sort_type = get_value('--sort', args)
    else:
        sort_type = 'split'

    model_dict = read_model_weights(base_path)
    sort_weights(base_path, model_dict, sort_type)
