"""
Convert GroupNet's NBA train.npy/test.npy into SocialCircle's dataset format
so that SocialCircle can train using GroupNet's exact train/test split.

GroupNet stores pre-sliced 15-frame windows in .npy files (feet).
SocialCircle expects per-clip ann.csv files + plist configs.

This script creates one clip per GroupNet sequence, capped at 32500 train
and 12500 test to match GroupNet's dataloader.

Usage:
    python scripts/convert_groupnet_nba.py

Then train with:
    python main.py --model evsc --split nba_groupnet \
        --obs_frames 5 --pred_frames 10
"""

import os
import plistlib

import numpy as np
from tqdm import tqdm

GROUPNET_TRAIN = '/Users/jdm/Documents/GroupNet/datasets/nba/train.npy'
GROUPNET_TEST = '/Users/jdm/Documents/GroupNet/datasets/nba/test.npy'
SC_ROOT = '/Users/jdm/Documents/SocialCircle'

DATASET_NAME = 'NBA_GroupNet'
SPLIT_NAME = 'nba_groupnet'

TRAIN_CAP = 32500
TEST_CAP = 12500

SCALE_FACTOR = 94.0 / 28.0

AGENT_NAMES = [f'player_{i}' for i in range(10)] + ['Ball']
AGENT_TYPES = ['TeamA'] * 5 + ['TeamB'] * 5 + ['Ball']
FRAME_IDS = list(range(0, 150, 10))


def write_clips(sequences, prefix, processed_dir, subsets_dir):
    """Write ann.csv and subset plist for each sequence."""
    clip_names = []
    for i in tqdm(range(len(sequences)), desc=prefix):
        name = f'{prefix}_{i:05d}'
        clip_names.append(name)

        clip_dir = os.path.join(processed_dir, name)
        os.makedirs(clip_dir, exist_ok=True)

        ann_path = os.path.join(clip_dir, 'ann.csv')
        positions = sequences[i]  # (15, 11, 2)
        lines = []
        for t, fid in enumerate(FRAME_IDS):
            for a in range(11):
                x, y = positions[t, a]
                lines.append(f'{fid},{AGENT_NAMES[a]},{x},{y},{AGENT_TYPES[a]}\n')
        with open(ann_path, 'w') as f:
            f.writelines(lines)

        plist_data = {
            'name': name,
            'dataset': DATASET_NAME,
            'annpath': f'./dataset_processed/{DATASET_NAME}/{name}/ann.csv',
            'order': [0, 1],
            'paras': [10, 25],
            'video_path': 'none',
            'matrix': [10.0, 0.0, 10.0, 0.0],
        }
        plist_path = os.path.join(subsets_dir, f'{name}.plist')
        with open(plist_path, 'wb') as f:
            plistlib.dump(plist_data, f)

    return clip_names


def main():
    train_raw = np.load(GROUPNET_TRAIN)[:TRAIN_CAP]
    test_raw = np.load(GROUPNET_TEST)[:TEST_CAP]

    train_data = train_raw / SCALE_FACTOR
    test_data = test_raw / SCALE_FACTOR

    print(f'Train sequences: {len(train_data)}')
    print(f'Test sequences:  {len(test_data)}')

    processed_dir = os.path.join(SC_ROOT, 'dataset_processed', DATASET_NAME)
    configs_dir = os.path.join(SC_ROOT, 'dataset_configs', DATASET_NAME)
    subsets_dir = os.path.join(configs_dir, 'subsets')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(subsets_dir, exist_ok=True)

    train_names = write_clips(train_data, 'gn_train', processed_dir, subsets_dir)
    test_names = write_clips(test_data, 'gn_test', processed_dir, subsets_dir)

    split_plist = {
        'dataset': DATASET_NAME,
        'type': 'meter',
        'scale': 1.0,
        'scale_vis': 1.0,
        'dimension': 2,
        'anntype': 'coordinate',
        'train': train_names,
        'test': test_names,
        'val': test_names,
    }
    split_path = os.path.join(configs_dir, f'{SPLIT_NAME}.plist')
    with open(split_path, 'wb') as f:
        plistlib.dump(split_plist, f)

    print(f'\nDone! Created {len(train_names)} train + {len(test_names)} test clips.')
    print(f'Split config: {split_path}')
    print(f'\nTrain with:')
    print(f'  python main.py --model evsc --split {SPLIT_NAME}'
          f' --obs_frames 5 --pred_frames 10')


if __name__ == '__main__':
    main()
