import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

    def forward(self, obs_emb):
        """
        Inputs:
        - obs_emb: tensor of shape (obs_len, batch, embedding_dim)
        Output:
        - final_h: tensor of shape (num_layers, batch, h_dim)
        """
        # Encode observed Trajectory
        batch = obs_emb.size(1)
        state_tuple = (
            torch.zeros(self.num_layers, batch, self.h_dim).to(obs_emb),
            torch.zeros(self.num_layers, batch, self.h_dim).to(obs_emb)
        )
        output, state = self.lstm(obs_emb, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=128, h_dim=64, num_layers=1, dropout=0.0):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, h_dim, num_layers, dropout=dropout)
        self.hidden2pos = MLP([h_dim, 2])

    def forward(self, last_pos, context, spatial_embedding):
        """
        Inputs:
        - last_pos: tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        Output:
        - pred_traj_fake: tensor of shape (seq_len, batch, 2)
        """
        pred_traj_fake = []
        pred_traj_fake_rel = []
        batch = last_pos.size(0)

        state_tuple = (context, context)

        for _ in range(self.seq_len):
            last_emb = spatial_embedding(last_pos)
            last_emb = last_emb.view(1, batch, -1)

            decoder_input = torch.cat([last_emb, context], dim=2)
            output, state_tuple = self.lstm(decoder_input, state_tuple)

            pos_rel = self.hidden2pos(output.view(-1, self.h_dim))
            last_pos = relative_to_abs(pos_rel.view(-1, batch, 2), last_pos)
            #last_pos = pos_rel
            last_pos = last_pos.view(batch, 2) # 1 x batch x 2 --> batch x 2

            pred_traj_fake.append(last_pos)
            pred_traj_fake_rel.append(pos_rel)

        pred_traj_fake = torch.stack(pred_traj_fake, dim=0)
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake, pred_traj_fake_rel


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

        self.spatial_embedding = MLP([2, embedding_dim])

        self.encoder = Encoder(embedding_dim=embedding_dim,
                               h_dim=encoder_h_dim,
                               num_layers=num_layers)

        self.decoder = Decoder(pred_len, embedding_dim+encoder_h_dim,
                               decoder_h_dim, num_layers)

        #self.transformer = MLP([encoder_h_dim, decoder_h_dim])

        self.apply(init_weights)

    def forward(self, obs_traj, pred_traj_gt=None):
        """
        Inputs:
        - obs_traj: tensor of shape (obs_len, batch, 2)
        - pred_traj_gt: tensor of shape (obs_len, batch, 2)
        Output:
        - pred_traj_fake: tensor of shape (pred_len, batch, 2)
        - loss
        """
        batch = obs_traj.size(1)
        obs_traj_nm = obs_traj - obs_traj[:,:,0].unsqueeze(2)
        last_pos = obs_traj_nm[-1] # batch x 2

        ##############################
        # Encode observed trajectory
        obs_emb = self.spatial_embedding(obs_traj_nm)
        obs_emb = obs_emb.view(-1, batch, self.embedding_dim)
        context = self.encoder(obs_emb)

        ##############################
        # Decoding
        pred_traj_fake = self.decoder(last_pos, context, self.spatial_embedding)

        ##############################
        # L2 loss
        if pred_traj_gt is not None:
            pred_traj_gt = pred_traj_gt - obs_traj[:,:,-1].unsqueeze(2)
            loss = l2_loss(pred_traj_gt, pred_traj_nm)
        else:
            loss = None

        pred_traj = pred_traj_nm + obs_traj[:,:,-1].unsqueeze(2)
        return pred_traj_fake, loss

    def grad_clipping(self, max_norm=0.25):
        nn.utils.clip_grad_norm_(self.encoder.lstm.parameters(), max_norm)
        nn.utils.clip_grad_norm_(self.decoder.lstm.parameters(), max_norm)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        depth = len(sizes)-1

        self.layers = nn.ModuleList([nn.Linear(sizes[0], sizes[1])])
        for i in range(1, depth):
            self.layers.extend([
                nn.BatchNorm1d(sizes[i]),
                nn.ReLU(inplace=True),
                nn.Linear(sizes[i], sizes[i+1])])

    def forward(self, input):
        out = input.contiguous().view(-1, input.size(-1))
        for layer in self.layers:
            out = layer(out)

        return out


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

    if mode == 'average':
        return torch.sqrt(loss.sum(dim=2)).mean()

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
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

