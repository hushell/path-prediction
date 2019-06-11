import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(2, embedding_dim) # TODO MLP

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: tensor of shape (num_layers, batch, h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = (
            torch.zeros(self.num_layers, batch, self.h_dim).to(obs_traj),
            torch.zeros(self.num_layers, batch, self.h_dim).to(obs_traj)
        )
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=64, h_dim=64,
                 num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim

        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        self.spatial_embedding = nn.Linear(2, embedding_dim) # TODO: MLP
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos_rel, state_tuple):
        """
        Inputs:
        - last_pos_rel: tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        Output:
        - pred_traj: tensor of shape (seq_len, batch, 2)
        """
        batch = last_pos_rel.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            decoder_input = self.spatial_embedding(rel_pos)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class TrajectoryPredictor(nn.Module):
    def __init__(self, obs_len, pred_len,
                 embedding_dim=64, encoder_h_dim=64, decoder_h_dim=64,
                 num_layers=1):
        super(TrajectoryPredictor, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.num_layers = num_layers

        self.encoder = Encoder(embedding_dim=embedding_dim,
                               h_dim=encoder_h_dim,
                               num_layers=num_layers)

        self.decoder = Decoder(pred_len, embedding_dim,
                               decoder_h_dim, num_layers)


    def forward(self, obs_traj_rel):
        """
        Inputs:
        - obs_traj_rel: tensor of shape (obs_len, batch, 2)
        Output:
        - pred_traj_rel: tensor of shape (pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)

        # Encode observed seq
        final_encoder_h = self.encoder(obs_traj_rel)
        final_encoder_h = final_encoder_h.view(-1, self.encoder_h_dim)

        decoder_h = final_encoder_h # TODO: MLP
        decoder_h = torch.unsqueeze(decoder_h, 0)
        decoder_c = torch.zeros(self.num_layers, batch,
                                self.decoder_h_dim).to(decoder_h)
        state_tuple = (decoder_h, decoder_c)

        # Predict trajectory
        last_pos_rel = obs_traj_rel[-1] # batch x 2
        pred_traj_fake_rel, final_decoder_h = self.decoder(last_pos_rel, state_tuple)

        return pred_traj_fake_rel


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    """
    Inputs:
    - dim_list: in the form of [input_dim, h_dim, ..., output_dim]
    """
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: tensor of shape (seq_len, batch, 2)
    - start_pos: tensor of shape (batch, 2)
    Outputs:
    - abs_traj: tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: tensor of shape (batch, seq_len, 1)
    - mode: sum, average or raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.float() * (pred_traj_gt.permute(1, 0, 2)
                                 - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

