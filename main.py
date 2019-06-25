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


#########################################################################
# Config

torch.backends.cudnn.benchmark = True

FORMAT = '[%(asctime)s %(levelname)s]: %(message)s'
logging.basicConfig(filename='print.log', filemode='w',
                    level=logging.INFO,
                    format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=16, type=int)
parser.add_argument('--pred_len', default=16, type=int)
parser.add_argument('--step', default=10, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--encoder_h_dim', default=64, type=int)
parser.add_argument('--decoder_h_dim', default=64, type=int)
parser.add_argument('--x_max', default=1630, type=int)
parser.add_argument('--y_max', default=1948, type=int)
parser.add_argument('--model_type', default='CNN', type=str)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=40, type=int)
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
device = torch.device('cuda:0') if args.use_gpu else torch.device('cpu')


#########################################################################
# Datasets

logger.info("Initializing train dataset")
train_set, train_loader = data_loader(args, 'train', 'Biker')
iterations_per_epoch = len(train_set) / args.batch_size
if args.num_epochs:
    args.num_iterations = int(iterations_per_epoch * args.num_epochs)
logger.info('There are {} iterations per epoch'.format(iterations_per_epoch))

logger.info("Initializing val dataset")
val_set, val_loader = data_loader(args, 'val', 'Biker')


#########################################################################
# Main
if args.model_type == 'CNN':
    from ConvNet import *
elif args.model_type == 'RNN':
    from LSTM import *

def main(args, train_loader, val_loader, device='cpu'):
    # Model
    predictor = TrajectoryPredictor(obs_len=args.obs_len,
                                    pred_len=args.pred_len,
                                    embedding_dim=args.embedding_dim,
                                    encoder_h_dim=args.encoder_h_dim,
                                    decoder_h_dim=args.decoder_h_dim,
                                    num_layers=args.num_layers).to(device)

    logger.info('Model structure:')
    logger.info(predictor)

    # Optimizier
    optimizer = optim.Adam(predictor.parameters(),
                           lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[15, 30],
                                               gamma=0.1)

    # Restore checkpoint
    checkpoint_path = os.path.join(args.output_dir,
                                   '%s_best.pt' % args.checkpoint_name)
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        predictor.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        t = checkpoint['t']
        epoch = checkpoint['epoch']
    else:
        t, epoch = 0, 0
        checkpoint = {
            'args': vars(args),
            'losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'metrics_ts': [],
        }

    # Main loop
    while t < args.num_iterations: # Only do a fixed number of iterations
        epoch += 1
        logger.info('=====> Starting epoch {} at iteration {}'.format(epoch, t+1))

        # lr scheduling
        scheduler.step()
        lr = scheduler.get_lr()[0]

        # One epoch
        tqdm_loader = tqdm(train_loader)
        for batch in tqdm_loader:

            # Train step
            losses = train_step(args, batch, predictor, optimizer)

            tqdm_loader.set_description('Epoch {}, Iter {} / {}, lr {}: losses = {}'.format(
                epoch, t+1, args.num_iterations, lr, losses))

            # Save losses
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t+1, args.num_iterations))
                for k, v in sorted(losses.items()):
                    logger.info('  [losses] {}: {:.3f}'.format(k, v))
                    checkpoint['losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Save checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                # Check stats on the validation set
                logger.info('Evaluation on val ...')
                metrics_val = evaluate(args, val_loader, predictor)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)

                # Check stats on the train set
                logger.info('Evaluation on train ...')
                metrics_train = evaluate(args, train_loader, predictor, limit=True)

                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                checkpoint['metrics_ts'].append(t)

                # Save metrics in log
                logname = os.path.join(args.output_dir, 'log.json')
                logger.info('Saving log to {}'.format(logname))
                with open(logname, 'w') as f:
                    checkpoint_sub = {k:v for k,v in checkpoint.items()
                                      if k not in {'model', 'optim', 't'}}
                    f.write(json.dumps(checkpoint_sub) + '\n')

                min_ade = min(checkpoint['metrics_val']['ade'])

                if metrics_val['ade'] == min_ade:
                    logger.info('Epoch %d: New lower avg_disp_error = %.4f' % (epoch, min_ade))
                    checkpoint['t'] = t
                    checkpoint['epoch'] = epoch
                    checkpoint['model'] = predictor.state_dict()
                    checkpoint['optim'] = optimizer.state_dict()

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
    Update model with a SGD step
    Outputs:
    - losses
    """
    predictor.train()
    losses = {}

    if args.use_gpu:
        batch = [tensor.to(device) for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
            obs_msk, pred_msk) = batch

    # Forward
    pred_traj_fake, loss = predictor.forward(obs_traj, pred_traj_gt)

    losses['l2_loss'] = loss.item()
    #loss = displacement_error(pred_traj_fake, pred_traj_gt, mode='average')
    #losses['ade_per_batch'] = loss.item()

    # Backward
    optimizer.zero_grad()
    loss.backward()
    predictor.grad_clipping(args.grad_max_norm)
    optimizer.step()

    return losses


def evaluate(args, loader, predictor, limit=False):
    """
    """
    predictor.eval()

    metrics = {}
    disp_error, f_disp_error = [], []
    total_traj = 0

    with torch.no_grad():
        for batch in loader:
            if args.use_gpu:
                batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                    obs_msk, pred_msk) = batch

            pred_traj_fake, loss = predictor.forward(obs_traj, pred_traj_gt)

            ade = displacement_error(pred_traj_fake, pred_traj_gt)
            fde = final_displacement_error(pred_traj_fake[:,:,-1], pred_traj_gt[:,:,-1])
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())

            total_traj += pred_traj_gt.size(1)

            if limit and total_traj >= args.num_samples_check:
                break

    ## DEBUG
    #for ii in range(5):
    #    print('==> [gt rel x=%.4f y=%.4f] [pred rel x=%.4f y=%.4f]' % (
    #        pred_traj_gt[ii,0,-1].item(), pred_traj_gt[ii,1,-1].item(),
    #        pred_traj_fake[ii,0,-1].item(), pred_traj_fake[ii,1,-1].item()))

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj

    predictor.train()
    return metrics


if __name__ == '__main__':
    predictor, checkpoint = main(args, train_loader, val_loader)
