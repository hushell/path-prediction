import torch
import torch.nn as nn


# input: batch_size * nc * isize
# output: batch_size * k * 1
class Encoder(nn.Module):
    def __init__(self, isize, nc, k=100, ndf=64):
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # input is nc x isize
        main = nn.Sequential()
        main.add_module('initial_conv_{0}-{1}'.format(nc, ndf),
                        nn.Conv1d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv1d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(out_feat),
                            nn.BatchNorm1d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Conv1d(cndf, k, 4, 1, 0, bias=False))

        self.main = main
        self.main.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output


# input: batch_size * k * 1
# output: batch_size * nc * isize
class Decoder(nn.Module):
    def __init__(self, isize, nc, k=100, ngf=64):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_convt'.format(k, cngf),
                        nn.ConvTranspose1d(k, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm1d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize = 4
        while csize < isize // 2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose1d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf // 2),
                            nn.BatchNorm1d(cngf // 2))
            main.add_module('pyramid_{0}_relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final_{0}-{1}_convt'.format(cngf, nc),
                        nn.ConvTranspose1d(cngf, nc, 4, 2, 1, bias=False))

        self.main = main
        self.main.apply(weights_init)

    def forward(self, input):
        output = self.main(input)
        return output


class TrajectoryPredictor(nn.Module):
    def __init__(self, obs_len, pred_len,
                 embedding_dim=32, encoder_h_dim=64, decoder_h_dim=64, num_layers=1):
        super(TrajectoryPredictor, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim

        self.encoder = Encoder(obs_len, 2, embedding_dim, encoder_h_dim)
        self.decoder = Decoder(pred_len, 2, embedding_dim, decoder_h_dim)

    def forward(self, obs_traj, pred_traj_gt=None):
        obs_traj_nm = obs_traj - obs_traj[:,:,0].unsqueeze(2)

        hidd = self.encoder(obs_traj_nm)
        pred_traj_nm = self.decoder(hidd)

        if pred_traj_gt is not None:
            pred_traj_gt = pred_traj_gt - obs_traj[:,:,-1].unsqueeze(2)
            loss = l2_loss(pred_traj_gt, pred_traj_nm)
        else:
            loss = None

        pred_traj = pred_traj_nm + obs_traj[:,:,-1].unsqueeze(2)
        return pred_traj, loss

    def grad_clipping(self, max_norm=0.25):
        return


def l2_loss(pred_traj, pred_traj_gt, loss_mask=None, mode='average'):
    """
    Input:
    - pred_traj: tensor of shape (batch, 2, seq_len). Predicted trajectory.
    - pred_traj_gt: tensor of shape (batch, 2, seq_len). Groud truth
    predictions.
    - loss_mask: tensor of shape (batch, 1, seq_len)
    - mode: sum, average or raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    if loss_mask:
        loss = loss_mask.float() * (pred_traj_gt - pred_traj)**2
    else:
        loss = (pred_traj_gt - pred_traj)**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        #return torch.sum(loss) / torch.numel(loss_mask.data)
        return loss.sum(dim=(1,2)).mean()
    elif mode == 'raw':
        return loss.sum(dim=(1,2))


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (batch, 2, seq_len). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (batch, 2, seq_len). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = (pred_traj_gt - pred_traj)**2

    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.mean(loss)
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
    loss = (pred_pos_gt - pred_pos)**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    elif mode == 'average':
        return torch.mean(loss)
    else:
        return torch.sum(loss)


def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)
