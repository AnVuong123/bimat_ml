import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torch.multiprocessing as mp
from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet
import random
from cgcnn.data import CIFDataMonoDB,CIFDataHetmono, get_train_val_test_loader, collate_pool_monodb, collate_pool_bidb, CIFDataBiDB, CIFDataHetDB, collate_pool_hetdb, collate_pool_hetmono
from cgcnn.model import BimonolayerCrystalGraphConvNet, HetDBlayerCrystalGraphConvNet, HetmonoCrystalGraphConvNet
parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
# parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
#                     help='dataset options, started with the path to root dir, '
#                          'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')


# parser.add_argument('--prop', choices=['bg', 'fermi',"cbm","vbm","e"],
#                     default='bg', help='target property to predict (default: band gap)')

parser.add_argument('--mono', choices=['1', '2'],
                    default='bg', help='use monolayer property (default: 1)')
                                                   
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-e' \
'poch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[500], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=1, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='Adam', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility across multiple libraries."""
    np.random.seed(seed)  # NumPy seed
    random.seed(seed)    # Python's built-in random module seed
    torch.manual_seed(seed) # PyTorch CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch CUDA seed for current GPU
        torch.cuda.manual_seed_all(seed) # PyTorch CUDA seed for all GPUs

    # Ensure deterministic behavior for CuDNN backend operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import pandas as pd

def run_hetmono(seed, gpu_id,k):
        
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Running seed {seed} on {device}")
        set_seed(seed)
        global args, best_mae_error
        best_mae_error = 1e10
        # load data
        #prop=args.prop
        prop="bg"
        if args.mono=="1":
            mono=True # use monolayer property
        else:
            mono=False # do not use monolayer property
        print(prop,mono)
        data_options=[f"data/reg-hetmono-train"]
        print(data_options)
        #args.train_size =len(df)
        #print(args.train_size)
        dataset = CIFDataHetmono(*data_options,k=k, classification =(args.task=='classification'), prop=prop)
        collate_fn = collate_pool_hetmono
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            return_test=True)

        # obtain target value normalizer
        if args.task == 'classification':
            normalizer = Normalizer(torch.zeros(2))
            normalizer.load_state_dict({'mean': 0., 'std': 1.})
        else:
            if len(dataset) < 500:
                warnings.warn('Dataset has less than 500 data points. '
                            'Lower accuracy is expected. ')
                sample_data_list = [dataset[i] for i in range(len(dataset))]
            else:
                sample_data_list = [dataset[i] for i in
                                    sample(range(len(dataset)), 500)]
            _,_, sample_target, _ , _, _,_,_,_ = collate_pool_hetmono(sample_data_list)
            normalizer = Normalizer(sample_target)

        # build model
        graph, graph2, _, _, _, _,_,_,_= dataset[0]
        atom, nbr, idx = graph
        atom2, nbr2, idx2 = graph2
        orig_atom_fea_len = atom.shape[-1]
        nbr_fea_len     = nbr.shape[-1]

        orig_atom_fea_len2 = atom2.shape[-1]
        nbr_fea_len2     = nbr2.shape[-1]

        model = HetmonoCrystalGraphConvNet(
                    orig_atom_fea_len,
                    orig_atom_fea_len2,
                    nbr_fea_len,
                    nbr_fea_len2,
                    atom_fea_len=args.atom_fea_len,
                    n_conv=args.n_conv,
                    h_fea_len=args.h_fea_len,
                    fusion_hidden=256,
                    classification=(args.task=='classification'),mono=mono)
        model.cuda()


        # define loss func and optimizer
        if args.task == 'classification':
            criterion = nn.NLLLoss()
            #criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.L1Loss()
        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_mae_error = checkpoint['best_mae_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                normalizer.load_state_dict(checkpoint['normalizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        finetune=False
        if finetune==True:
            checkpoint_path="pre-trained/band-gap.pth.tar"
            checkpoint=torch.load(checkpoint_path,map_location="cpu")
            # --- Load model weights ---
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            normalizer.load_state_dict(checkpoint["normalizer"])
            #start_epoch = checkpoint['epoch']
            #best_mae_error = checkpoint['best_mae_error']
        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                                gamma=0.5)
    
        for epoch in range(args.start_epoch, args.epochs):
            print(args.epochs)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{args.epochs}] - Current LR: {current_lr:.6f}")
            # train for one epoch
            train_hetmono(train_loader, model, criterion, optimizer, epoch, normalizer,mono)

            # evaluate on validation set
            mae_error = 0 #validate(val_loader, model, criterion, normalizer)

            if mae_error != mae_error:
                print('Exit due to NaN')
                sys.exit(1)

            scheduler.step()

            # remember the best mae_eror and save checkpoint
            if args.task == 'regression':
                is_best = mae_error < best_mae_error
                best_mae_error = min(mae_error, best_mae_error)
            else:
                is_best = mae_error > best_mae_error
                best_mae_error = max(mae_error, best_mae_error)
            if (epoch + 1) % 500 == 0:
                print(prop,mono)
                if mono==True:
                    last_ckpt_path = f'pre-trained/model_last_hetmono_{prop}_mono{prop}_{epoch+1}ep_{args.lr}_128b_kfold_{k}.pth.tar'
                else:
                    last_ckpt_path = f'pre-trained/model_last_hetmono_{prop}_{epoch+1}ep_{args.lr}_128b_kfold_{k}.pth.tar'
                torch.save({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_mae_error': best_mae_error,
                    'optimizer': optimizer.state_dict(),
                    'normalizer': normalizer.state_dict(),
                    'args': vars(args)
                }, last_ckpt_path)
                print(f"=> Saved last epoch {epoch+1} model to {last_ckpt_path}")
            #save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'best_mae_error': best_mae_error,'optimizer': optimizer.state_dict(),'normalizer': normalizer.state_dict(),'args': vars(args)}, is_best)

        # test best model
        #last_ckpt_path = f'pre-trained/model_last_hetmono_bg_0.5_500ep_128b_kfold_{k}.pth.tar'
        #torch.save({
        #    'epoch': args.epochs,
        #    'state_dict': model.state_dict(),
        #    'best_mae_error': best_mae_error,
        #    'optimizer': optimizer.state_dict(),
        #    'normalizer': normalizer.state_dict(),
        #    'args': vars(args)
        #}, last_ckpt_path)
        #print(f"=> Saved last epoch model to {last_ckpt_path}")
        print('---------Evaluate Model on Test Set---------------')
        #best_checkpoint = torch.load('pre-trained/model_best.pth.tar')
        #model.load_state_dict(best_checkpoint['state_dict'])
        #validate(test_loader, model, criterion, normalizer, test=False)

                
def train_hetmono(train_loader, model, criterion, optimizer, epoch, normalizer,mono):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, input2, target, mono_target1,mono_target2, _, _,s_vector, l_vector) in enumerate(train_loader):

     
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            input_var2 = (Variable(input2[0].cuda(non_blocking=True)),
                         Variable(input2[1].cuda(non_blocking=True)),
                         input2[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input2[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
            input_var2 = (Variable(input2[0]),
                         Variable(input2[1]),
                         input2[2],

                         input2[3])
        #mono_target1,mono_target2 = mono_target1.cuda(non_blocking=True).view(-1, 1), mono_target2.cuda(non_blocking=True).view(-1, 1)
        mono_target1 = mono_target1.cuda(non_blocking=True)
        mono_target2 = mono_target2.cuda(non_blocking=True)
        s_vector = s_vector.cuda(non_blocking=True)
        l_vector = l_vector.cuda(non_blocking=True)
        # ================================
        # 3) Normalize target
        # ================================
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()

        if args.cuda:
            target_var = target_normed.cuda(non_blocking=True)
        else:
            target_var = target_normed

        # ================================
        # 4) Forward pass
        # ================================
        #print(s_vector.shape,l_vector.shape)
        output = model(*input_var,*input_var2, s_vector,l_vector, mono_target1,mono_target2,mono=mono)
        loss = criterion(output, target_var)

        # ================================
        # 5) Metrics
        # ================================
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.cpu()), target)
            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # ================================
        # 6) Backprop
        # ================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ================================
        # 7) Logging
        # ================================
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')
            else:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Accu {accuracies.val:.3f} ({accuracies.avg:.3f})\t'
                      f'Precision {precisions.val:.3f} ({precisions.avg:.3f})\t'
                      f'Recall {recalls.val:.3f} ({recalls.avg:.3f})\t'
                      f'F1 {fscores.val:.3f} ({fscores.avg:.3f})\t'
                      f'AUC {auc_scores.val:.3f} ({auc_scores.avg:.3f})')
                


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='pre-trained/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'pre-trained/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    list_seed = [16,16,16,16]
    num_gpus = torch.cuda.device_count()
    print("GPUs available:", num_gpus)

    processes = []
    k=0
    mp.set_start_method('spawn', force=True)
    for idx, seed in enumerate(list_seed):
        gpu_id = idx % num_gpus
        p = mp.Process(target=run_hetmono, args=(seed, gpu_id, k))
        p.start()
        processes.append(p)
        k=k+1

    for p in processes:
        p.join()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()

