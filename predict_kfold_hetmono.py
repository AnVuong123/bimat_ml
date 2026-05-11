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
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser(description='Crystal gated neural networks')
# parser.add_argument('modelpath', help='path to the trained model.')
# parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--prop', choices=['bg', 'fermi',"cbm","vbm","e"],
                    default='bg', help='target property to predict (default: band gap)')

parser.add_argument('--mono', choices=['1', '2'],
                    default='bg', help='use monolayer property (default: 1)')

args = parser.parse_args(sys.argv[1:])
# if os.path.isfile(model_path):
#     print("=> loading model params '{}'".format(model_path))
#     model_checkpoint = torch.load(model_path,
#                                   map_location=lambda storage, loc: storage)
#     model_args = argparse.Namespace(**model_checkpoint['args'])
#     print("=> loaded model params '{}'".format(model_path))
# else:
#     print("=> no model params found at '{}'".format(model_path))

args.cuda = not args.disable_cuda and torch.cuda.is_available()

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

set_seed(16)


def main_hetmono():
    
    global args, model_args, best_mae_error
    
    # load data
    
    list_accuracy=[]
    list_precision=[]
    list_recall=[]
    list_f1=[]
    list_mae=[]
    list_mse=[]
    list_rmse=[]
    list_r2=[]
    c=0
    list_seed=[16]
    prop=args.prop
    mono=False
    if args.mono=="1":
        mono=True # use monolayer properties
    else:
        mono=False # do not use monolayer properties
    for k in range(0,4):
        set_seed(list_seed[0])
        print(k)
        model_path=f"pre-trained/model_hetmono_kfold"
        model_path=f"{model_path}_{k}.pth.tar"
        model_checkpoint = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
       
        cifpath=f"data/reg-hetmono-test"
        dataset = CIFDataHetmono(cifpath,k=k, prop=prop)
        collate_fn = collate_pool_hetmono
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, collate_fn=collate_fn,
                                pin_memory=args.cuda)
        print(len(dataset),len(test_loader))
        for i, batch in enumerate(test_loader):
            print(f"\nBatch {i} type:", type(batch))
            if isinstance(batch, (list, tuple)):
                for j, part in enumerate(batch):
                    t = type(part)
                    shape = getattr(part, "shape", None)
                    size0 = getattr(part, "size", lambda *_: None)(0) if hasattr(part, "size") else None
                    print(f"  part[{j}]: type={t}, shape={shape}, size0={size0}, len={len(part) if hasattr(part,'__len__') else None}")
        
        batch = next(iter(test_loader))
       
        # build model
        graph, graph2, _, _, _, _, _,_, _= dataset[0]
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
                    atom_fea_len=model_args.atom_fea_len,
                    n_conv=model_args.n_conv,
                    h_fea_len=model_args.h_fea_len,
                    fusion_hidden=258,
                    classification=(model_args.task=='classification'),mono=mono)
        model.cuda()
        if args.cuda:
            model.cuda()
    
        # define loss func and optimizer
        if model_args.task == 'classification':
            criterion = nn.NLLLoss()
            #criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.L1Loss()
        # if args.optim == 'SGD':
        #     optimizer = optim.SGD(model.parameters(), args.lr,
        #                           momentum=args.momentum,
        #                           weight_decay=args.weight_decay)
        # elif args.optim == 'Adam':
        #     optimizer = optim.Adam(model.parameters(), args.lr,
        #                            weight_decay=args.weight_decay)
        # else:
        #     raise NameError('Only SGD or Adam is allowed as --optim')

        normalizer = Normalizer(torch.zeros(3))

        # optionally resume from a checkpoint
        if os.path.isfile(model_path):
            print("=> loading model '{}'".format(model_path))
            checkpoint = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded model '{}' (epoch {}, validation {})"
                .format(model_path, checkpoint['epoch'],
                        checkpoint['best_mae_error']))
        else:
            print("=> no model found at '{}'".format(model_path))

        #validate(test_loader, model, criterion, normalizer, test=True)
        #embeds, ids = validate_embeddings(test_loader, model, use_cuda=args.cuda,save_path=f'val_mono_embeddings_100ep_128b_{2}.csv',l2_normalize=True)
        embeddings=False
        if model_args.task == 'classification':
            c=1
            accuracy, precision, recall,f1=validate_hetmono(test_loader, model, criterion, normalizer,c,mono, test=True)
            list_accuracy.append(accuracy)
            list_precision.append(precision)
            list_recall.append(recall)
            list_f1.append(f1)
        else:
            c=0
            if embeddings==True:
                embeds, ids = validate_embeddings(test_loader, model, use_cuda=args.cuda,
                                                  save_path=f'val_full_mono_500_embeddings_{k}.csv',l2_normalize=True)
            else:
                mae, mse, rmse, r2=validate_hetmono(test_loader, model, criterion, normalizer, c, mono, test=True)
                list_mae.append(mae)
                list_mse.append(mse)
                list_rmse.append(rmse)
                list_r2.append(r2)

    if model_args.task == 'classification':
        print("========== Average Classification Metrics ==========")
        print(model_path)
        print(list_f1)
        print(f"Accuracy : {np.mean(list_accuracy):.4f} ± {np.std(list_accuracy):.4f}")
        print(f"Precision: {np.mean(list_precision):.4f} ± {np.std(list_precision):.4f}")
        print(f"Recall   : {np.mean(list_recall):.4f} ± {np.std(list_recall):.4f}")
        print(f"F1-score : {np.mean(list_f1):.4f} ± {np.std(list_f1):.4f}")
    else:
        print("========== Average Regression Metrics ==========")
        print(model_path)
        print(list_mae)
        print(f"MAE  : {np.mean(list_mae):.4f} ± {np.std(list_mae):.2f}")
        print(f"MSE  : {np.mean(list_mse):.4f} ± {np.std(list_mse):.2f}")
        print(f"RMSE : {np.mean(list_rmse):.4f} ± {np.std(list_rmse):.2f}")
        print(f"R2   : {np.mean(list_r2):.4f} ± {np.std(list_r2):.2f}")  

        

      

def validate_embeddings(val_loader, model, use_cuda=True, save_path='val_embeddings.csv', l2_normalize=True):
    model.eval()
    batch_time = AverageMeter()

    all_ids = []
    all_embeds = []

    end = time.time()
    with torch.no_grad():
        for i, (input, target, batch_cif_ids) in enumerate(val_loader):
            # move to device
            if use_cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])


            reps = model.encode(*input_var)  # (batch_size, D)
            all_embeds.append(reps.cpu())
            all_ids += batch_cif_ids
            batch_time.update(time.time() - end)
            end = time.time()

       

    embeds = torch.cat(all_embeds, dim=0).numpy()  # (N_graphs, D)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mono_name'] + [f'c_{k}' for k in range(embeds.shape[1])])
        for cid, vec in zip(all_ids, embeds):
            writer.writerow([cid] + vec.tolist())

    print(f'Wrote {len(all_ids)} embeddings to {save_path}')
    return embeds, all_ids




    

def validate_hetmono(val_loader, model, criterion, normalizer, c, mono=False, test=False):
    import time, csv, torch
    from torch.autograd import Variable

    batch_time  = AverageMeter()
    losses      = AverageMeter()

    if model_args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls    = AverageMeter()
        fscores    = AverageMeter()
        auc_scores = AverageMeter()

    test_targets, test_preds, test_cif_ids = [], [], []

    print("len_val_loader =", len(val_loader))
    model.eval()

    end = time.time()

    for i, (input, input2, target, mono_target1,mono_target2, cif_id, cif_id2,s_vector, l_vector) in enumerate(val_loader):

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

        # normalize target
        if model_args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()

        target_var = target_normed.cuda(non_blocking=True) if args.cuda else target_normed

        # ===============================
        # Forward bilayer model
        # ===============================
        with torch.no_grad():
            output = model(*input_var,*input_var2,s_vector,l_vector, mono_target1,mono_target2,mono)
            loss = criterion(output, target_var)

        # ===============================
        # Regression metrics
        # ===============================
        if model_args.task == 'regression':
            output_denorm = (normalizer.denorm(output.cpu()))

            mae_error = mae(output_denorm, target)

            losses.update(loss.item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))

            if test:
                test_preds   += output_denorm.view(-1).tolist()
                test_targets += target.view(-1).tolist()
                for cid1, cid2 in zip(cif_id, cif_id2):
                    test_cif_ids.append(f"{cid1}_{cid2}")

        else:
            # ===============================
            # Classification metrics
            # ===============================
            probs = torch.softmax(output.cpu(), dim=1)
            target_cpu = target_var.cpu()

            accuracy, precision, recall, fscore, auc_score = \
                class_eval(probs, target_cpu)

            losses.update(loss.item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

            if test:
                pred_labels = probs.argmax(dim=1)
                test_preds   += pred_labels.tolist()
                test_targets += target_cpu.view(-1).tolist()
                for cid1, cid2 in zip(cif_id, cif_id2):
                    test_cif_ids.append(f"{cid1}_{cid2}")

        # ===============================
        # Timing prints
        batch_time.update(time.time() - end)
        end = time.time()

        if model_args.task == 'regression':
            print('Val: [{0}/{1}] '
                  'Time {bt.val:.3f} ({bt.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'MAE {mae.val:.3f} ({mae.avg:.3f})'
                  .format(i, len(val_loader),
                          bt=batch_time, loss=losses, mae=mae_errors))
        else:
            print('Val: [{0}/{1}] '
                  'Time {bt.val:.3f} ({bt.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Acc {acc.val:.3f} ({acc.avg:.3f})  '
                  'Prec {prec.val:.3f} ({prec.avg:.3f})  '
                  'Rec {rec.val:.3f} ({rec.avg:.3f})  '
                  'F1 {f1.val:.3f} ({f1.avg:.3f})  '
                  'AUC {auc.val:.3f} ({auc.avg:.3f})'
                  .format(i, len(val_loader),
                          bt=batch_time, loss=losses,
                          acc=accuracies, prec=precisions, rec=recalls,
                          f1=fscores, auc=auc_scores))

    # ======================================================
    # FINAL RETURN FOR TEST MODE
    # ======================================================
    if c == 0:   # regression
        print("len test_cif_ids =", len(test_cif_ids))
        print("len test_targets =", len(test_targets))
        print("len test_preds   =", len(test_preds))
        df = pd.DataFrame({
            "cif_pair": test_cif_ids,
            "target": test_targets,
            "pred": test_preds,
        })
        df.to_csv("test_results_r.csv", index=False)

        mae1  = mean_absolute_error(df["target"], df["pred"])
        mse1  = mean_squared_error(df["target"], df["pred"])
        rmse1 = np.sqrt(mse1)
        r21   = r2_score(df["target"], df["pred"])
        return mae1, mse1, rmse1, r21

    else:  # classification
        df = pd.DataFrame({
            "cif_pair": test_cif_ids,
            "target": test_targets,
            "pred": test_preds,
        })
        df.to_csv("test_results_c.csv", index=False)

        acc  = accuracy_score(df["target"], df["pred"])
        prec = precision_score(df["target"], df["pred"], zero_division=0)
        rec  = recall_score(df["target"], df["pred"])
        f1   = f1_score(df["target"], df["pred"])
        return acc, prec, rec, f1
    
    


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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    start = time.time()
    main_hetmono()
    end = time.time()

    print(f"TOTAL TIME: {end - start:.4f} seconds")
