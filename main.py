import argparse
import logging
import os
import json
import sys
import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import data_loader
from models import *


#########################################################################
# config
torch.backends.cudnn.benchmark = True

FORMAT = '[%(asctime)s %(levelname)s]: %(message)s'
logging.basicConfig(filename='print.log', filemode='w',
                    level=logging.INFO,
                    format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=20, type=int)
parser.add_argument('--pred_len', default=20, type=int)
parser.add_argument('--step', default=10, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--encoder_h_dim', default=64, type=int)
parser.add_argument('--decoder_h_dim', default=64, type=int)
parser.add_argument('--x_max', default=1630, type=int)
parser.add_argument('--y_max', default=1948, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--grad_max_norm', default=1.0, type=float)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=30, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--gpu_id', default="0", type=str)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

#########################################################################
# datasets
logger.info("Initializing train dataset")
train_set, train_loader = data_loader(args, 'train', 'Pedestrian')
iterations_per_epoch = len(train_set) / args.batch_size
if args.num_epochs:
    args.num_iterations = int(iterations_per_epoch * args.num_epochs)
logger.info('There are {} iterations per epoch'.format(iterations_per_epoch))

logger.info("Initializing val dataset")
val_set, val_loader = data_loader(args, 'val', 'Pedestrian')


#########################################################################
# main loop
def main(args, train_loader, val_loader):
    long_dtype, float_dtype = get_dtypes(args)
    device = torch.device('cuda:0') if args.use_gpu else torch.device('cpu')

    # Model
    predictor = TrajectoryPredictor(obs_len=args.obs_len,
                                    pred_len=args.pred_len,
                                    embedding_dim=args.embedding_dim,
                                    encoder_h_dim=args.encoder_h_dim,
                                    decoder_h_dim=args.decoder_h_dim,
                                    num_layers=args.num_layers).to(device)
    #predictor = TrajectoryLSTM(obs_len=args.obs_len,
    #                           pred_len=args.pred_len,
    #                           embedding_dim=args.embedding_dim,
    #                           h_dim=args.encoder_h_dim,
    #                           num_layers=args.num_layers).to(device)


    logger.info('Model structure:')
    logger.info(predictor)

    # Optimizier
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,60], gamma=0.1)

    # Main loop
    t, epoch = 0, 0

    checkpoint = {
        'args': vars(args),
        'losses': defaultdict(list),
        'losses_ts': [],
        'metrics_val': defaultdict(list),
        'metrics_train': defaultdict(list),
        'metrics_ts': [],
    }

    while t < args.num_iterations: # Only do a fixed number of iterations
        epoch += 1

        logger.info('=====> Starting epoch {} at iteration {}'.format(epoch, t+1))
        scheduler.step()
        lr = scheduler.get_lr()[0]
        tqdm_loader = tqdm(train_loader)
        for batch in tqdm_loader:
            # train step
            losses = train_step(args, batch, predictor, optimizer)
            tqdm_loader.set_description('Epoch {}, Iter {} / {}, lr {}: losses = {}'.format(
                epoch, t+1, args.num_iterations, lr, losses))

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t+1, args.num_iterations))
                for k, v in sorted(losses.items()):
                    logger.info('  [losses] {}: {:.3f}'.format(k, v))
                    checkpoint['losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                # Check stats on the validation set
                logger.info('Evaluation on val ...')
                metrics_val = check_accuracy(args, val_loader, predictor)
                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)

                # Check stats on the train set
                logger.info('Evaluation on train ...')
                metrics_train = check_accuracy(args, train_loader, predictor, limit=True)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                checkpoint['metrics_ts'].append(t)

                # save metrics in log
                logname = os.path.join(args.output_dir, 'log.json')
                logger.info('Saving log to {}'.format(logname))
                with open(logname, 'w') as f:
                    checkpoint_sub = {k:v for k,v in checkpoint.items()
                                      if k not in {'model_state', 'optim_state', 'best_t'}}
                    f.write(json.dumps(checkpoint_sub) + '\n')

                min_ade = min(checkpoint['metrics_val']['ade'])

                if metrics_val['ade'] == min_ade:
                    logger.info('Epoch %d: New lower avg_disp_error = %.4f' % (epoch, min_ade))
                    checkpoint['best_t'] = t
                    checkpoint['model_state'] = predictor.state_dict()
                    checkpoint['optim_state'] = optimizer.state_dict()

                    checkpoint_path = os.path.join(args.output_dir,
                            '%s_best.pt' % (args.checkpoint_name))
                    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    torch.save(checkpoint, checkpoint_path)

            t += 1
            if t >= args.num_iterations:
                print('=====> Terminate training at epoch {} iteration {}'.format(epoch, t+1))
                break

    return predictor, checkpoint

def train_step(args, batch, predictor, optimizer):
    """
    Outputs:
    - losses
    """
    predictor.train()
    losses = {}

    if args.use_gpu:
        batch = [tensor.to(device) for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
            obs_msk, pred_msk) = batch

    pred_traj_fake, pred_traj_fake_rel = predictor.forward(obs_traj,
            args.x_max, args.y_max, pred_traj_gt)

    loss = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, pred_msk, mode='average')
    losses['l2_loss_rel'] = loss.item()
    loss = l2_loss(pred_traj_fake, pred_traj_gt, pred_msk, mode='average')
    losses['l2_loss_abs'] = loss.item()
    #loss = displacement_error(pred_traj_fake, pred_traj_gt, mode='average')
    #losses['ade-batch'] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    predictor.grad_clipping(args.grad_max_norm)
    optimizer.step()

    return losses


def check_accuracy(args, loader, predictor, limit=False):
    """
    """
    predictor.eval()

    metrics = {}
    l2_losses_abs, l2_losses_rel = [], []
    disp_error, f_disp_error = [], []
    total_traj = 0
    loss_mask_sum = 0

    with torch.no_grad():
        for batch in loader:
            if args.use_gpu:
                batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                    obs_msk, pred_msk) = batch

            pred_traj_fake, pred_traj_fake_rel = predictor(obs_traj,
                    args.x_max, args.y_max, pred_traj_gt)

            #l2_loss_abs, l2_loss_rel = cal_l2_losses(
            #        pred_traj_gt, pred_traj_gt_rel,
            #        pred_traj_fake, pred_traj_fake_rel, pred_msk)

            ade = displacement_error(pred_traj_fake, pred_traj_gt)
            fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
            #l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, pred_msk, mode='sum')
            #l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, pred_msk, mode='sum')

            disp_error.append(ade.item())
            f_disp_error.append(fde.item())
            #l2_losses_abs.append(l2_loss_abs.item())
            #l2_losses_rel.append(l2_loss_rel.item())

            loss_mask_sum += torch.numel(pred_msk)
            total_traj += pred_traj_gt.size(1)

            if limit and total_traj >= args.num_samples_check:
                break

    # DEBUG
    for ii in range(5):
        print('==> [gt rel x=%.4f y=%.4f] [pred rel x=%.4f y=%.4f]' % (
            pred_traj_gt[-1,ii,0].item(), pred_traj_gt[-1,ii,1].item(),
            pred_traj_fake[-1,ii,0].item(), pred_traj_fake[-1,ii,1].item()))

    #metrics['l2_loss_abs'] = sum(l2_losses_abs) / loss_mask_sum
    #metrics['l2_loss_rel'] = sum(l2_losses_rel) / loss_mask_sum
    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj

    predictor.train()
    return metrics


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel,
                  pred_traj_fake, pred_traj_fake_rel,
                  loss_mask):
    l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')

    return l2_loss_abs, l2_loss_rel


if __name__ == '__main__':
    predictor, checkpoint = main(args, train_loader, val_loader)
