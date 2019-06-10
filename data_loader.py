from torch.utils.data import DataLoader

import logging
import os
import math
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def data_loader(args, split='train', agent_label='Biker'):
    dset = StanfordDroneDataset(split, agent_label,
                                obs_len=args.obs_len,
                                pred_len=args.pred_len,
                                step=args.step)

    loader = DataLoader(dset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.loader_num_workers)
                        #collate_fn=seq_collate)
    return dset, loader


class StanfordDroneDataset(Dataset):
    def __init__(self, split, agent_label, obs_len=20, pred_len=20, step=10):
        """
        Data format: <track_id> <x> <y> <frame_id> <msk> <agent_label>
        Inputs:
        - split: train or val
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - step: Number of frames to skip while making the dataset
        """
        super(StanfordDroneDataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.step = step
        self.seq_len = self.obs_len + self.pred_len

        # read annotation.txt
        data = self.read_file(agent_label)

        # collect sequences
        track_ids = np.unique([line[0] for line in data]) # all track ids
        if split == 'train':
            track_ids = track_ids[:int(0.9*len(track_ids))]
        elif split == 'val':
            track_ids = track_ids[int(0.9*len(track_ids)):]
        else:
            raise ValueError('No such split!')

        self.seq_list = [] # list, each seq is seq_len x 2
        self.seq_list_rel = [] # same as seq_list, relative movement to previous location
        self.loss_mask_list = [] # list, each is seq_len x 1

        for id in track_ids:
            lines_of_id = [line for line in data if line[0] == id]

            # process each sequence (sample seq with gap ``step'')
            for t in range(0, len(lines_of_id) - self.seq_len + 1, step):
                lines_for_seq = lines_of_id[t:t + self.seq_len]
                assert(len(lines_for_seq) == self.seq_len)
                seq = np.vstack([line[1:3] for line in lines_for_seq]) # seq_len x 2
                self.seq_list.append(seq)
                rel_seq = np.zeros(seq.shape)
                rel_seq[1:, :] = seq[1:, :] - seq[:-1, :]
                self.seq_list_rel.append(rel_seq)
                self.loss_mask_list.append( np.vstack([line[4] for line in lines_for_seq]) )

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        '''
        Outputs:
        - obs_seq, pred_seq, obs_seq_rel, pred_seq_rel, obs_msk, pred_msk
        '''
        out = [self.seq_list[index][:self.obs_len, :], self.seq_list[index][self.obs_len:, :],
               self.seq_list_rel[index][:self.obs_len, :], self.seq_list_rel[index][self.obs_len:, :],
               self.loss_mask_list[index][:self.obs_len, :], self.loss_mask_list[index][self.obs_len:, :]]
        return out

    def read_file(self, agent_label):
        '''
        Read file with format
        <track_id> <x_min> <y_min> <x_max> <y_max> <frame_id> <lost> <occlud> <sth> <agent_type> <sth>
        and transfer to the format <track_id> <x_min> <y_min> <frame_id> <msk>.
        NB: x_min = x_max, y_min = y_max
        Input:
        - agent_label: Pedestrian or Biker or Car
        Output:
        - data: list of [<track_id> <x> <y> <frame_id> <msk>]
        '''
        data = []
        delim = ' '
        path = './annotations.txt'

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split(delim)
                if line[9] != agent_label:
                    continue
                line = [int(line[0]), float(line[1]), float(line[2]), int(line[5]),
                        int(bool(line[6]) | bool(line[7]))]
                data.append(line)
        return data


def seq_collate(data):
    (obs_seq, pred_seq, obs_seq_rel, pred_seq_rel,
            obs_mask, pred_mask) = zip(*data)

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq, dim=0).permute(1, 0, 2)
    pred_traj = torch.cat(pred_seq, dim=0).permute(1, 0, 2)
    obs_traj_rel = torch.cat(obs_seq_rel, dim=0).permute(1, 0, 2)
    pred_traj_rel = torch.cat(pred_seq_rel, dim=0).permute(1, 0, 2)
    obs_mask = torch.cat(obs_mask, dim=0)
    pred_mask = torch.cat(pred_mask, dim=0)

    out = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
           obs_mask, pred_mask]

    return tuple(out)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def plot_seq(img, seq_ped, color, thickness=15, line_type=8):
    '''
    Input:
    - img: reference image
    - seq_ped: np array of shape (traj_len, 2)
    '''
    seq_ped = seq_ped.astype(int)
    for start, end in zip(seq_ped[:-1], seq_ped[1:]):
        cv2.line(img, tuple(start), tuple(end),
                 color, thickness, line_type, shift=0)
