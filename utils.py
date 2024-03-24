import torch
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
import numpy as np
from scipy.sparse import identity, diags


def graph2adj(adj):
    #hat_adj = adj + identity(adj.shape[0])
    hat_adj = adj
    degree_vec = hat_adj.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
    d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
    degree_matrix  = diags(d_inv_sqrt, 0)
    DAD = degree_matrix @ (hat_adj @ degree_matrix)
    AD = hat_adj @ (degree_matrix @ degree_matrix)
    DA = degree_matrix @ (degree_matrix @ hat_adj)

    return DAD, AD, DA

def preprocess(data, args):
    print("Preprocessing node features!!!!!!")
    nnodes = data.x.shape[0]
    if args.undirected:
        data.edge_index = to_undirected(data.edge_index, nnodes)
        row, col = data.edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(nnodes, nnodes))
        adj = adj.to_scipy(layout='csr')
        DAD, AD, DA = graph2adj(adj)
        norm_adj = SparseTensor.from_scipy(DAD).float()
        feat_lst = []
        feat_lst.append(data.x)
        high_order_features = data.x.clone()
        for _ in range(args.num_hops):
            high_order_features = norm_adj @ high_order_features
            #data.x = torch.cat((data.x, high_order_features), dim=1)
            feat_lst.append(high_order_features)
        data.x = torch.stack(feat_lst, dim=1)
        #data.num_features *= (1+args.num_hops)
    else:
        row, col = data.edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(nnodes, nnodes))
        adj = adj.to_scipy(layout='csr')
        _, _, DA = graph2adj(adj)
        _, _, DA_tran = graph2adj(adj.transpose())
        norm_adj = SparseTensor.from_scipy(DA).float()
        norm_adj_tran = SparseTensor.from_scipy(DA_tran).float()
        feat_lst = []
        feat_lst.append(data.x)
        high_order_features = data.x.clone()
        high_order_features_tran = data.x.clone()
        for _ in range(args.num_hops):
            high_order_features = norm_adj @ high_order_features
            high_order_features_tran = norm_adj @ high_order_features_tran
            #data.x = torch.cat((data.x, high_order_features, high_order_features_tran), dim=1)
            feat_lst.append(high_order_features)
            feat_lst.append(high_order_features_tran)
        data.x = torch.stack(feat_lst, dim=1)
        #data.num_features *= (1+2*args.num_hops)

    return data

def all_numpy(obj):
    # Ensure everything is in numpy or int or float (no torch tensor)

    if isinstance(obj, dict):
        for key in obj.keys():
            all_numpy(obj[key])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            all_numpy(obj[i])
    else:
        if not isinstance(obj, (np.ndarray, int, float)):
            return False

    return True

