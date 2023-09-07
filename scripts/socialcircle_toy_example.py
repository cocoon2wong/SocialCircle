"""
@Author: Conghao Wong
@Date: 2023-07-12 17:38:42
@LastEditors: Conghao Wong
@LastEditTime: 2023-09-07 09:06:53
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import TK_BORDER_WIDTH, TK_TITLE_STYLE, TextboxHandler

sys.path.insert(0, os.path.abspath('.'))
import qpid
from qpid.utils import dir_check, get_mask
from main import main

DATASET = 'ETH-UCY'
SPLIT = 'zara1'
CLIP = 'zara1'
MODEL_PATH = 'weights/Silverbullet/SocialCircle_zara1'

TEMP_IMG_PATH = './temp_files/socialcircle_toy_example/fig.png'
LOG_PATH = './temp_files/socialcircle_toy_example/run.log'

dir_check(os.path.dirname(LOG_PATH))


class BetaToyExample():
    def __init__(self, args: list[str]) -> None:
        self.t: qpid.training.Structure = None
        self.image: tk.PhotoImage = None

        self.inputs: list[tf.Tensor] = None
        self.outputs: list[tf.Tensor] = None
        self.input_and_gt: list[list[tf.Tensor]] = None

        self.load_model(args)
        
    def init_model(self):
        self.t.model = self.t.create_model()
        self.t.agent_manager.set_types(self.t.model.input_types,
                                       self.t.label_types)
        
        if self.input_and_gt is None:
            self.input_and_gt = \
                list(self.t.agent_manager.make(CLIP, 'test'))

    def load_model(self, args: list[str]) -> qpid.training.Structure:
        try:
            t = main(args, run_train_or_test=False)
            self.t = t
            self.init_model()
            self.t.log(
                f'Model `{toy.t.args.loada}` and dataset files ({CLIP}) loaded.')
        except Exception as e:
            print(e)

    def run_on_agent(self, agent_index: int,
                     extra_neighbor_position=None):

        inputs = self.input_and_gt[agent_index][:-1]
        inputs = [i[tf.newaxis] for i in inputs]

        if (p := extra_neighbor_position) is not None:
            nei = self.add_one_neighbor(inputs, p)
            inputs[1] = nei

        self.forward(inputs)
        self.draw_results()

    def add_one_neighbor(self, inputs: list[tf.Tensor],
                         position: list[tuple[float, float]]):
        '''
        Shape of `nei` should be `(1, max_agents, obs, 2)`
        '''
        obs = inputs[0]
        nei = inputs[1]

        nei = nei.numpy()
        steps = nei.shape[-2]

        xp = np.array([0, steps-1])
        fp = np.array(position)
        x = np.arange(steps)

        traj = np.column_stack([np.interp(x, xp, fp[:, 0]),
                                np.interp(x, xp, fp[:, 1])])

        nei_count = self.get_neighbor_count(nei)
        nei[0, nei_count] = traj - obs.numpy()[0, -1:, :]
        return tf.cast(nei, tf.float32)

    def forward(self, inputs: list[tf.Tensor]):
        self.inputs = inputs
        self.outputs = self.t.model.forward(inputs, training=False)

    def get_neighbor_count(self, neighbor_obs: tf.Tensor):
        '''
        Input's shape should be `(1, max_agents, obs, dim)`.
        '''
        nei = neighbor_obs[0]
        nei_mask = get_mask(tf.reduce_sum(nei, axis=[-1, -2]))
        return int(tf.reduce_sum(nei_mask))

    def draw_results(self):
        inputs = self.inputs
        outputs = self.outputs

        obs = inputs[0][0].numpy()      # (obs, dim)
        nei = inputs[1][0].numpy()      # (max_agents, obs, dim)
        out = outputs[0][0].numpy()

        c_obs = self.t.picker.get_center(obs)
        c_nei = self.t.picker.get_center(nei)
        c_out = self.t.picker.get_center(out)

        plt.figure()

        # draw neighbors
        nei_count = self.get_neighbor_count(inputs[1])
        _nei = c_nei[:nei_count, :, :] + c_obs[np.newaxis, -1, :]
        plt.plot(_nei[:, -1, 0], _nei[:, -1, 1], 'o', 
                 color='darkorange', markersize=13)

        # draw neighbors' trajectories
        _nei = np.reshape(_nei, [-1, 2])
        plt.plot(_nei[:, 0], _nei[:, 1], 's', color='purple')

        # draw observations
        plt.plot(c_obs[:, 0], c_obs[:, 1], 's', color='cornflowerblue')

        # draw predictions
        for pred in c_out:
            plt.plot(pred[:, 0], pred[:, 1], 's')

        plt.axis('equal')
        save_dir = os.path.dirname(TEMP_IMG_PATH)
        dir_check(save_dir)
        plt.savefig(TEMP_IMG_PATH)
        plt.close()
        self.image = tk.PhotoImage(file=TEMP_IMG_PATH)


def run_prediction(t: BetaToyExample,
                   agent_id: tk.StringVar,
                   px0: tk.StringVar,
                   py0: tk.StringVar,
                   px1: tk.StringVar,
                   py1: tk.StringVar,
                   canvas: tk.Label,
                   social_circle: tk.Label,
                   nei_angles: tk.Label):

    if (px0 and py0 and px0 and py0 and
        len(x0 := px0.get()) and len(y0 := py0.get()) and
            len(x1 := px1.get()) and len(y1 := py1.get())):
        extra_neighbor = [[float(x0), float(y0)],
                          [float(x1), float(y1)]]
        t.t.log('Start running with an addition neighbor' +
                f'from {extra_neighbor[0]} to {extra_neighbor[1]}...')
    else:
        extra_neighbor = None
        t.t.log('Start running without any manual inputs...')

    t.run_on_agent(int(agent_id.get()),
                   extra_neighbor_position=extra_neighbor)
    canvas.config(image=t.image)
    time = int(1000 * t.t.model.inference_times[-1])
    t.t.log(f'Running done. Time cost = {time} ms.')

    # Set numpy format
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})

    # SocialCircle
    sc = t.outputs[1][1].numpy()[0]
    social_circle.config(text=str(sc.T))

    # All neighbors' angles
    count = t.get_neighbor_count(t.inputs[1])
    na = t.outputs[1][2].numpy()[0][:count]
    nei_angles.config(text=str(na*180/np.pi))


if __name__ == '__main__':

    root = tk.Tk()
    root.title('Toy Example of SocialCircle Models')

    # Left column
    l_args = {
        # 'background': '#FFFFFF',
        'border': TK_BORDER_WIDTH,
    }

    left_frame = tk.Frame(root, **l_args)
    left_frame.grid(row=0, column=0, sticky=tk.NW)

    tk.Label(left_frame, text='Settings',
             **TK_TITLE_STYLE, **l_args).grid(
                 column=0, row=0, sticky=tk.W)

    agent_id = tk.StringVar(left_frame, '1195')
    tk.Label(left_frame, text='Agent ID', **l_args).grid(
        column=0, row=1)
    tk.Entry(left_frame, textvariable=agent_id).grid(
        column=0, row=2)

    px0 = tk.StringVar(left_frame)
    tk.Label(left_frame, text='New Neighbor (x-axis, start)', **l_args).grid(
        column=0, row=3)
    tk.Entry(left_frame, textvariable=px0).grid(
        column=0, row=4)

    py0 = tk.StringVar(left_frame)
    tk.Label(left_frame, text='New Neighbor (y-axis, start)', **l_args).grid(
        column=0, row=5)
    tk.Entry(left_frame,  textvariable=py0).grid(
        column=0, row=6)

    px1 = tk.StringVar(left_frame)
    tk.Label(left_frame, text='New Neighbor (x-axis, end)', **l_args).grid(
        column=0, row=7)
    tk.Entry(left_frame, textvariable=px1).grid(
        column=0, row=8)

    py1 = tk.StringVar(left_frame)
    tk.Label(left_frame, text='New Neighbor (y-axis, end)', **l_args).grid(
        column=0, row=9)
    tk.Entry(left_frame,  textvariable=py1).grid(
        column=0, row=10)

    # Right Column
    r_args = {
        'background': '#FFFFFF',
        'border': TK_BORDER_WIDTH,
    }
    t_args = {
        'foreground': '#000000',
    }

    right_frame = tk.Frame(root, **r_args)
    right_frame.grid(row=0, column=1, sticky=tk.NW, rowspan=2)

    tk.Label(right_frame, text='Predictions',
             **TK_TITLE_STYLE, **r_args, **t_args).grid(
                 column=0, row=0, sticky=tk.W)
    
    tk.Label(right_frame, text='Model Path:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        column=0, row=1)
    (model_path := tk.Label(right_frame, width=60, wraplength=510, 
                            text=MODEL_PATH, **r_args, **t_args)).grid(
        column=1, row=1)

    tk.Label(right_frame, text='Social Circle:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        column=0, row=2)
    (sc := tk.Label(right_frame, width=60, **r_args, **t_args)).grid(
        column=1, row=2)

    tk.Label(right_frame, text='Neighbor Angles:', width=16, anchor=tk.E, **r_args, **t_args).grid(
        column=0, row=3)
    (angles := tk.Label(right_frame, width=60, **r_args, **t_args)).grid(
        column=1, row=3)

    tk.Canvas(right_frame, width=640, height=480, **r_args).grid(
        column=0, row=4, columnspan=2)
    (canvas := tk.Label(right_frame, **r_args, **t_args)).grid(
        column=0, row=4, columnspan=2)

    # Log frame
    log_frame = tk.Frame(right_frame, **r_args)
    log_frame.grid(column=0, row=5, columnspan=2)

    logbar = tk.Text(log_frame, width=89, height=7, **r_args, **t_args)
    (scroll := tk.Scrollbar(log_frame, command=logbar.yview)).pack(
        side=tk.RIGHT, fill=tk.Y)
    logbar.config(yscrollcommand=scroll.set)
    logbar.pack()

    # Init model and training structure
    def args(path): return ['main.py', '--model', 'MKII',
                            '--loada', path,
                            '--loadb', 'speed',
                            '--force_dataset', DATASET,
                            '--force_split', SPLIT,
                            '--force_clip', CLIP]

    qpid.set_log_path(LOG_PATH)
    qpid.set_log_stream_handler(TextboxHandler(logbar))
    toy = BetaToyExample(args(MODEL_PATH))

    # Button Frame
    b_args = {
        # 'background': '#FFFFFF',
        # 'border': TK_BORDER_WIDTH,
    }

    button_frame = tk.Frame(root, **b_args)
    button_frame.grid(column=0, row=1, sticky=tk.N)

    tk.Button(button_frame, text='Run Prediction',
              command=lambda: run_prediction(
                  toy, agent_id, px0, py0, px1, py1,
                  canvas, sc, angles), **b_args).grid(
        column=0, row=10, sticky=tk.N)

    tk.Button(button_frame, text='Run Prediction (original)',
              command=lambda: run_prediction(
                  toy, agent_id, None, None, None, None,
                  canvas, sc, angles), **b_args).grid(
        column=0, row=11, sticky=tk.N)

    tk.Button(button_frame, text='Reload Model Weights',
              command=lambda: [toy.load_model(args(p := filedialog.askdirectory(initialdir=os.path.dirname(MODEL_PATH)))),
                               model_path.config(text=p)]).grid(
        column=0, row=12, sticky=tk.N)

    root.mainloop()
