import argparse

import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LayerNorm
import torch_geometric.transforms as T

from logger import Logger
from tqdm import tqdm
import os
import sys
import numpy as np
import pandas as pd
import time
import copy

from utils import preprocess
from multihead_att import MultiheadAtt
from dataset_prep import PygNodePropPredDataset, Evaluator


#torch.set_num_threads(80)
torch.manual_seed(0)
# for original mult
# 1042 for 2
# 42 for 6, 4
# 999, 0 for 8
# 99999 for 10

# for tech mapped mult
# 0 for 16
#10 for 12
# 99, 126374 for 20
# 666,66 for 24
# 77 for 28

class HOGA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_hops, heads):
        super(HOGA, self).__init__()
        self.num_layers = num_layers
        self.num_hops = num_hops
        use_bias = False

        self.lins = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()
        self.trans = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        attn_drop = 0.0
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels, bias=use_bias))
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.gates.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.trans.append(MultiheadAtt(hidden_channels, heads, dropout=attn_drop))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=use_bias))
            self.gates.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=use_bias))
            self.trans.append(MultiheadAtt(hidden_channels, heads, dropout=attn_drop))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))

        # two linear layer for predictions
        self.linear = torch.nn.ModuleList()
        #self.linear.append(Linear(self.num_hops*hidden_channels, hidden_channels, bias=False))
        self.linear.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))

        self.bn0 = BatchNorm1d(hidden_channels)
        self.attn_layer = torch.nn.Linear(2 * hidden_channels, 1)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for gate in self.gates:
            gate.reset_parameters()
        for li in self.linear:
            li.reset_parameters()
        self.bn0.reset_parameters()

    def forward(self, x):
        x = self.lins[0](x)
        #x = F.relu(x)

        for i, tran in enumerate(self.trans):
            x = self.lns[i](self.gates[i](x)*(tran(x, x, x)[0]))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        target = x[:,0,:].unsqueeze(1).repeat(1,self.num_hops-1,1)
        split_tensor = torch.split(x, [1, self.num_hops-1], dim=1)
        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        x = (node_tensor + neighbor_tensor).squeeze()
        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x1 = self.linear[1](x) # for xor
        x2 = self.linear[2](x) # for maj
        x3 = self.linear[3](x) # for roots
        #return x, x1.log_softmax(dim=-1), x2.log_softmax(dim=-1), x3.log_softmax(dim=-1)
        return x1, x2, x3, layer_atten


def train(model, train_loader, optimizer, device, args):
    model.train()
    total_loss = 0
    start_time = time.time()
    for i, (x, y, r_y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        optimizer.zero_grad()
        out1, out2, out3, attn = model(x)

        ### build labels for multitask
        ### original 0: PO, 1: plain, 2: shared, 3: maj, 4: xor, 5: PI
        y1 = y.squeeze(1).clone().detach() # make (maj and xor) as xor
        for i in range(y1.size()[-1]):
            if y1[i] == 0 or y1[i] == 5:
                y1[i] = 1
            if y1[i] == 2:
                y1[i] = 4
            if y1[i] > 2:
                y1[i] = y1[i] - 1 # make to 5 classes
            y1[i] = y1[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor

        y2 = y.squeeze(1).clone().detach() # make (maj and xor) as maj
        for i in range(y2.size()[-1]):
            if y2[i] > 2:
                y2[i] = y2[i] - 1 # make to 5 classes
            if y2[i] == 0 or y2[i] == 4:
                y2[i] = 1
            y2[i] = y2[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor

        # for root classification
        # 0: PO, 1: maj, 2: xor, 3: and, 4: PI
        # y3 = data_r.y.squeeze(1)[n_id[:batch_size]]
        y3 = r_y.squeeze(1).clone().detach()
        for i in range(y3.size()[-1]):
            if y3[i] == 0 or y3[i] == 4:
                y3[i] = 3
            y3[i] = y3[i] - 1 # 3 classes: 0: maj, 1: xor, 2: and+PI+PO


        loss = F.cross_entropy(out1, y1) + args.lda1*F.cross_entropy(out2, y2) + args.lda2*F.cross_entropy(out3, y3)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_time = time.time()-start_time
    loss = total_loss / len(train_loader)

    return loss, train_time

def post_processing(out1, out2):
    pred_1 = out1.argmax(dim=-1, keepdim=True)
    pred_2 = out2.argmax(dim=-1, keepdim=True)
    pred_ecc = (out1 + out2).argmax(dim=-1, keepdim=True)

    l =  pred_1.size()[0]
    pred = []
    for i in range(l):
        if pred_1[i] == pred_2[i]:
            if pred_1[i] == 0: # PO, and, PI
                pred.append(torch.tensor([1]))
            else: # maj, xor
                pred.append(pred_1[i] + 2) # 3 or 4
        else:
            if (pred_1[i] == 1 and pred_2[i] == 2) or (pred_1[i] == 2 and pred_2[i] == 1):
                pred.append(torch.tensor([2])) # maj and xor
            else:
                if pred_ecc[i] == 0: # PO, and, PI
                    pred.append(torch.tensor([1]))
                else: # maj, xor
                    pred.append(pred_ecc[i] + 2)
    pred = torch.tensor(pred)
    '''
    pred = copy.deepcopy(pred_1)

    eq_idx = (torch.eq(pred_1, pred_2) == True).nonzero(as_tuple=True)[0]
    # if pred_1[i] != 0  # maj, xor
    eq_mx_idx = (pred_1[eq_idx] != 0).nonzero(as_tuple=True)[0]
    # pred_1[i] = pred_1[i] + 2  -->  3, 4
    pred[eq_idx[eq_mx_idx]] = pred_1[eq_idx[eq_mx_idx]] + 2
    # if pred_1[i] == 0 PI/PI/and --> final 1
    eq_aig_idx = (pred_1[eq_idx] == 0).nonzero(as_tuple=True)[0]
    pred[eq_idx[eq_aig_idx]] = 1

    neq_idx = (torch.eq(pred_1, pred_2) == False).nonzero(as_tuple=True)[0]
    # if pred_1[i] == 1 and pred_2[i] == 2 shared --> 2
    p1 = (pred_1[neq_idx] == 2).nonzero(as_tuple=True)[0]
    p2 = (pred_2[neq_idx] == 1).nonzero(as_tuple=True)[0]
    shared = p1[(p1.view(1, -1) == p2.view(-1, 1)).any(dim=0)]
    pred[neq_idx[shared]] = 2

    p1 = (pred_1[neq_idx] == 1).nonzero(as_tuple=True)[0]
    p2 = (pred_2[neq_idx] == 2).nonzero(as_tuple=True)[0]
    shared = p1[(p1.view(1, -1) == p2.view(-1, 1)).any(dim=0)]
    pred[neq_idx[shared]] = 2

    # else (error correction for discrepant predictions)
    if len(p1) != len(p2) or len(p1) != len(neq_idx):
        print("start error correction!!!!!!")
        v, freq = torch.unique(torch.cat((p1, p2), 0), sorted=True, return_inverse=False, return_counts=True, dim=None)
        uniq = (freq == 1).nonzero(as_tuple=True)[0]
        ecc = v[uniq]
        ecc_mx = (pred_ecc[neq_idx][ecc] != 0).nonzero(as_tuple=True)[0]
        ecc_aig = (pred_ecc[neq_idx][ecc] == 0).nonzero(as_tuple=True)[0]
        pred[neq_idx[ecc[ecc_mx]]] = pred_ecc[neq_idx][ecc][ecc_mx] + 2
        pred[neq_idx[ecc[ecc_aig]]] = 1
        zz = (pred == 0).nonzero(as_tuple=True)[0]
        pred[zz] = 1
    '''

    return torch.reshape(pred, (pred.shape[0], 1))


@torch.no_grad()
def test(model, train_loader, valid_loader, test_loader, device):
    model.eval()

    train_acc_r = 0
    train_acc_s = 0
    valid_acc_r = 0
    valid_acc_s = 0
    test_acc_r = 0
    test_acc_s = 0
    for i, (x, y, r_y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        out1, out2, out3, train_attn = model(x)
        y_pred_shared = post_processing(out1, out2).to(device)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)

        y_shared = y.squeeze(1).clone().detach()
        y_root = r_y.squeeze(1).clone().detach()
        # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1
        # 0: maj, 1: xor, 2: and+PI+PO
        r0 = (y_root == 0).nonzero(as_tuple=True)[0]
        r4 = (y_root == 4).nonzero(as_tuple=True)[0]
        y_root[r0] = 3
        y_root[r4] = 3
        y_root = y_root - 1
        y_root = torch.reshape(y_root, (y_root.shape[0], 1))
        y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))

        train_acc_r += y_pred_root.eq(y_root).double().sum()
        train_acc_s += y_pred_shared.eq(y_shared).double().sum()

    train_acc_r /= len(train_loader.dataset)
    train_acc_s /= len(train_loader.dataset)

    for i, (x, y, r_y) in enumerate(valid_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        out1, out2, out3, valid_attn = model(x)
        y_pred_shared = post_processing(out1, out2).to(device)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)

        y_shared = y.squeeze(1).clone().detach()
        y_root = r_y.squeeze(1).clone().detach()
        # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1
        # 0: maj, 1: xor, 2: and+PI+PO
        r0 = (y_root == 0).nonzero(as_tuple=True)[0]
        r4 = (y_root == 4).nonzero(as_tuple=True)[0]
        y_root[r0] = 3
        y_root[r4] = 3
        y_root = y_root - 1
        y_root = torch.reshape(y_root, (y_root.shape[0], 1))
        y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))

        valid_acc_r += y_pred_root.eq(y_root).double().sum()
        valid_acc_s += y_pred_shared.eq(y_shared).double().sum()

    valid_acc_r /= len(valid_loader.dataset)
    valid_acc_s /= len(valid_loader.dataset)

    start_time = time.time()
    for i, (x, y, r_y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        out1, out2, out3, test_attn = model(x)
        y_pred_shared = post_processing(out1, out2).to(device)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)

        y_shared = y.squeeze(1).clone().detach()
        y_root = r_y.squeeze(1).clone().detach()
        # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1
        # 0: maj, 1: xor, 2: and+PI+PO
        r0 = (y_root == 0).nonzero(as_tuple=True)[0]
        r4 = (y_root == 4).nonzero(as_tuple=True)[0]
        y_root[r0] = 3
        y_root[r4] = 3
        y_root = y_root - 1
        y_root = torch.reshape(y_root, (y_root.shape[0], 1))
        y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))

        test_acc_r += y_pred_root.eq(y_root).double().sum()
        test_acc_s += y_pred_shared.eq(y_shared).double().sum()
    inference_time = time.time() - start_time
    #print('The inference time is %s' % inference_time)

    test_acc_r /= len(test_loader.dataset)
    test_acc_s /= len(test_loader.dataset)

    print(train_attn.shape)
    print(valid_attn.shape)
    print(test_attn.shape)

    return train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s, inference_time


@torch.no_grad()
def test_all(model, test_loader, device, file_name=None):
    model.eval()

    test_acc_r = 0
    test_acc_s = 0
    start_time = time.time()
    all_y_shared = []
    all_pred_shared = []
    all_test_attn = []
    for i, (x, y, r_y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        out1, out2, out3, test_attn = model(x)
        y_pred_shared = post_processing(out1, out2).to(device)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)

        y_shared = y.squeeze(1).clone().detach()
        y_root = r_y.squeeze(1).clone().detach()
        # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1
        # 0: maj, 1: xor, 2: and+PI+PO
        r0 = (y_root == 0).nonzero(as_tuple=True)[0]
        r4 = (y_root == 4).nonzero(as_tuple=True)[0]
        y_root[r0] = 3
        y_root[r4] = 3
        y_root = y_root - 1
        y_root = torch.reshape(y_root, (y_root.shape[0], 1))
        y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))

        test_acc_r += y_pred_root.eq(y_root).double().sum()
        test_acc_s += y_pred_shared.eq(y_shared).double().sum()

        all_y_shared.append(y_shared.detach().cpu().squeeze().numpy())
        all_pred_shared.append(y_pred_shared.detach().cpu().squeeze().numpy())
        all_test_attn.append(test_attn.detach().cpu().squeeze().numpy())
    inference_time = time.time() - start_time
    print('The inference time is %s' % inference_time)

    test_acc_r /= len(test_loader.dataset)
    test_acc_s /= len(test_loader.dataset)

    return 0, 0, test_acc_r, 0, 0, test_acc_s, inference_time

def main():
    parser = argparse.ArgumentParser(description='mult16')
    parser.add_argument('--bits', type=int, default=64)
    parser.add_argument('--bits_test', type=int, default=64)
    parser.add_argument('--datagen', type=int, default=0,
		help="0=multiplier generator, 1=adder generator, 2=loading design")
    # (0)(1) require bits as inputs; (2) requires designfile as input
    parser.add_argument('--datagen_test', type=int, default=0,
		help="0=multiplier generator, 1=adder generator, 2=loading design")
    # (0)(1) require bits as inputs; (2) requires designfile as input
    parser.add_argument('--multilabel', type=int, default=1,
        help="0=5 classes; 1=6 classes with shared xor/maj as a new class; 2=multihot representation")
    parser.add_argument('--num_class', type=int, default=6)
    parser.add_argument('--designfile', '-f', type=str, default='')
    parser.add_argument('--designfile_test', '-ft', type=str, default='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=80)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr', type=float, default=0.008)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_hops', type=int, default=10)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--design_copies', type=int, default=1)
    parser.add_argument('--mapped', type=int, default=0)
    parser.add_argument('--lda1', type=int, default=1)
    parser.add_argument('--lda2', type=int, default=2)
    parser.add_argument('--design', type=str, default='booth')
    parser.add_argument('--undirected', action='store_true')
    parser.add_argument('--test_all_bits', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = torch.device('cpu') ## cpu for now only

    if not os.path.exists(f'models/'):
        os.makedirs(f'models/')

    ## generate dataset
    root_folder = os.path.abspath((os.path.dirname(os.path.abspath("gnn_multitask.py"))))
    print(root_folder)
    # new generator functionalities: 0) multiplier 1) adder 2) read design

    if args.datagen == 0:
        prefix = 'mult'
    elif args.datagen == 1:
        prefix = 'adder'
    if args.mapped == 1:
        suffix ="_7nm_mapped"
    elif args.mapped == 2:
        suffix ="_mapped"
    else:
        suffix = ''
    #suffix = ''

    if args.design == "booth":
        design_name = "booth_" + prefix + str(args.bits) + suffix
        root_path = "/scratch-x3/circuit_datasets/booth/"
    else:
        design_name = prefix + str(args.bits) + suffix
        root_path = "/scratch-x3/circuit_datasets/csa/"
        #root_path = "/export/scratch-x2/circuit_datasets/csa/"
    train_design_name = design_name
    design_name_root = design_name + "_root"
    design_name_shared = design_name + "_shared"

    ### training dataset loading
    master = pd.read_csv('/work/shared/users/phd/cd574/transformer/my_gamora/Gamora/graph_transformer-gamora/dataset_prep/master.csv', index_col = 0)
    if not design_name_root in master:
        os.system(f"python dataset_prep/make_master_file.py --design_name {design_name_root}")
    if not design_name_shared in master:
        os.system(f"python dataset_prep/make_master_file.py --design_name {design_name_shared}")
    dataset_r = PygNodePropPredDataset(name=f'{design_name_root}', root=root_path)
    print("Training on %s" % design_name)
    data_r = dataset_r[0]
    #data_r = preprocess(data_r, args)
    data_r = T.ToSparseTensor()(data_r)

    dataset = PygNodePropPredDataset(name=f'{design_name_shared}', root=root_path)
    data = dataset[0]
    data = preprocess(data, args)
    data = T.ToSparseTensor()(data)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']#.to(device)
    valid_idx = split_idx['valid']#.to(device)
    test_idx = split_idx['test']#.to(device)

    batch_data_train = Data.TensorDataset(data.x[train_idx], data.y[train_idx], data_r.y[train_idx])
    #batch_data_valid = Data.TensorDataset(data.x[valid_idx], data.y[valid_idx], data_r.y[valid_idx])
    batch_data_test = Data.TensorDataset(data.x[test_idx], data.y[test_idx], data_r.y[test_idx])

    train_loader = Data.DataLoader(batch_data_train, batch_size=args.batch_size, shuffle=True, num_workers=10)
    #valid_loader = Data.DataLoader(batch_data_valid, batch_size=args.batch_size, shuffle=False, num_workers=10)
    test_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=False, num_workers=10)

    model = HOGA(data.num_features, args.hidden_channels,
                     3, args.num_layers,
                     args.dropout, num_hops=args.num_hops+1, heads=args.heads).to(device)

    #data_r = data_r.to(device)
    #data = data.to(device)

    logger_r = Logger(args.runs, args)
    logger = Logger(args.runs, args)

    train_time = 0
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
        best_test_r = float('-inf')
        best_test_s = float('-inf')
        for epoch in range(1, 1 + args.epochs):
            loss, epoch_train_time = train(model, train_loader, optimizer, device, args)
            #scheduler.step()
            train_time += epoch_train_time
            # train val
            #result = test(model, train_loader, valid_loader, test_loader, device)
            result = test_all(model, test_loader, device)
            logger_r.add_result(run, result[:3])
            logger.add_result(run, result[3:-1])

            if epoch % args.log_steps == 0:
                train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s, _= result
                if test_acc_s >= best_test_s:
                    best_test_r = test_acc_r
                    best_test_s = test_acc_s
                    if args.save_model:
                        model_name = f'models/hoga_{design_name}_{args.design}.pt'
                        torch.save({'model_state_dict': model.state_dict()}, model_name)
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'[Root Model] Train: {100 * train_acc_r:.2f}%, '
                      f'[Root Model] Valid: {100 * valid_acc_r:.2f}% '
                      f'[Root Model] Test: {100 * test_acc_r:.2f}% '
                      f'[Shared Model] Train: {100 * train_acc_s:.2f}%, '
                      f'[Shared Model] Valid: {100 * valid_acc_s:.2f}% '
                      f'[Shared Model] Test: {100 * test_acc_s:.2f}%')

        logger_r.print_statistics(run)
        logger.print_statistics(run)
    logger_r.print_statistics()
    logger.print_statistics()
    train_time /= args.runs

    ### evaluation dataset loading
    logger_eval_r = Logger(1, args)
    logger_eval = Logger(1, args)

    if args.datagen == 0:
        prefix = 'mult'
    elif args.datagen == 1:
        prefix = 'adder'
    if args.mapped == 1:
        suffix ="_7nm_mapped"
    elif args.mapped == 2:
        suffix ="_mapped"
    else:
        suffix = ''

    if args.test_all_bits:
        #bits_test_lst = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768]
        bits_test_lst = [64, 128, 192, 256, 320, 384]
    else:
        bits_test_lst = [args.bits_test]

    ## load pre-trained model
    if args.save_model:
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])

    for bits_test in bits_test_lst:
        if args.design == "booth":
            design_name = "booth_" + prefix + str(bits_test) + suffix
        else:
            design_name = prefix + str(bits_test) + suffix
        design_name_root = design_name + "_root"
        design_name_shared = design_name + "_shared"
        print("Evaluation on %s" % design_name)

        master = pd.read_csv('/work/shared/users/phd/cd574/transformer/my_gamora/Gamora/graph_transformer-gamora/dataset_prep/master.csv', index_col = 0)
        if not design_name_root in master:
            os.system(f"python dataset_prep/make_master_file.py --design_name {design_name_root}")
        if not design_name_shared in master:
            os.system(f"python dataset_prep/make_master_file.py --design_name {design_name_shared}")
        dataset_r = PygNodePropPredDataset(name=f'{design_name_root}', root=root_path)
        data_r = dataset_r[0]
        #data_r = preprocess(data_r, args)
        data_r = T.ToSparseTensor()(data_r)

        dataset = PygNodePropPredDataset(name=f'{design_name_shared}', root=root_path)
        data = dataset[0]
        data = preprocess(data, args)
        data = T.ToSparseTensor()(data)

        # tensor placement
        #data_r.y = data_r.y.to(device)
        #data = data.to(device)

        batch_data_test = Data.TensorDataset(data.x, data.y, data_r.y)
        test_loader = Data.DataLoader(batch_data_test, batch_size=args.batch_size, shuffle=False, num_workers=10)

        for run_1 in range(1):
            for epoch in range(1):
                file_name = f'{args.design}_{design_name_shared}'
                result = test_all(model, test_loader, device, file_name)
                logger_eval_r.add_result(run_1, result[:3])
                logger_eval.add_result(run_1, result[3:-1])
                if epoch % args.log_steps == 0:
                    train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s, inference_time = result
                    print(f'Run: {run_1 + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'[Root Model] Train: {100 * train_acc_r:.2f}%, '
                          f'[Root Model] Valid: {100 * valid_acc_r:.2f}% '
                          f'[Root Model] Test: {100 * test_acc_r:.2f}% '
                          f'[Shared Model] Train: {100 * train_acc_s:.2f}%, '
                          f'[Shared Model] Valid: {100 * valid_acc_s:.2f}% '
                          f'[Shared Model] Test: {100 * test_acc_s:.2f}%')

            # logger_eval_r.print_statistics(run_1)
            # logger_eval.print_statistics(run_1)

        logger_eval_r.print_statistics()
        logger_eval.print_statistics()

        ## save results
        if not os.path.exists(f'results/hoga'):
            os.makedirs(f'results/hoga')
        filename = f'results/hoga/{args.design}_{train_design_name}.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(
                f"{design_name} " + f"{args.weight_decay} " + f"{args.dropout} " + f"{args.lr} " + \
                f"{args.num_layers} " + f"{args.epochs} " + f"{args.hidden_channels} " + \
                f"train_time: {train_time:.4f} " + f"inference_time: {inference_time:.4f} " + \
                f"test_acc_r: {100 * test_acc_r:.2f} " + f"test_acc_s: {100 * test_acc_s:.2f} \n")

if __name__ == "__main__":
    main()
