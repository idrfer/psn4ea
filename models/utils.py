import os 
import gc
import time
import math
import scipy
import torch
import random
import multiprocessing

from network import *

import numpy as np
import scipy.sparse as sp
import torch.optim as optim

from torch.utils.data import Dataset

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

def csls_sim(sim_mat, k):
    nearest_values1 = torch.mean(torch.topk(sim_mat, k)[0], 1)
    nearest_values2 = torch.mean(torch.topk(sim_mat.t(), k)[0], 1)
    csls_sim_mat = 2 * sim_mat.t() - nearest_values1
    csls_sim_mat = csls_sim_mat.t() - nearest_values2
    return csls_sim_mat

def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def run_args(args):
    seed = args.seed
    if seed==0:
        seed = int(time.time())%114514
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    return device

def run_version(args):
    split        = ""
    img_vec_path = ""
    file_dir     = args.file_dir
    if   "V1" in file_dir:
        split = "norm"
        img_vec_path = "datasets/pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
    elif "V2" in file_dir:
        split = "dense"
        img_vec_path = "datasets/pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
    else:
        split = file_dir.split("/")[-1]
        img_vec_path = "datasets/pkls/"+ split+ "_GA_id_img_feature_dict.pkl"
    return split, img_vec_path

def read_raw_data(file_dir, l=[1,2]):

    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups
    
    def read_dict(file_paths):
        ids         = {}
        ent2id_dict = {}
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids
    
    print("load raw data from ", file_dir)
    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in l])
    ills             = read_file([file_dir + "/ill_ent_ids"])
    triples          = read_file([file_dir + "/triples_" + str(i) for i in l])

    r_hs, r_ts = {}, {}
    for (h,r,t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return ent2id_dict, ills, triples, r_hs, r_ts, ids

def get_topk_indices(M, K=1000):
    _, W   = M.shape
    M_view = M.view(-1)
    vals, indices = M_view.topk(K)
    indices_2d = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
    return indices_2d

def get_adjr(ent_size, triples, norm=False):
    M = {}
    for tri in triples:
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 0
        if tri[0] == tri[2]:
            continue
        M[(tri[0], tri[2])] += 1
    ind_tem, val_tem = [], []
    for (fir, sec) in M:
        ind_tem.append((fir, sec))
        ind_tem.append((sec, fir))
        val_tem.append(M[(fir, sec)])
        val_tem.append(M[(fir, sec)])
    for i in range(ent_size):
        ind_tem.append((i, i))
        val_tem.append(1)

    if not norm:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind_tem).t(),\
                                     torch.FloatTensor(val_tem), torch.Size([ent_size, ent_size]))
    else:
        ind_tem = np.array(ind_tem, dtype=np.int32)
        val_tem = np.array(val_tem, dtype=np.float32)
        adj     = sp.coo_matrix((val_tem, (ind_tem[:, 0], ind_tem[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        M = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    return M

def get_model(m_name, args, device):
    n_units = [int(x) for x in args.hidden_units.strip().split(",")]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    if m_name == "GAT":
        model = GAT(n_units, n_heads, args.dropout, args.attn_dropout, args.instance_normalization, True).to(device)
    else:
        model = GCN(n_units, args.dropout).to(device)
    return model

def get_ucl(device, tau, Lambda, n_view):
    return ucl_loss(device, tau, Lambda, n_view, 1, False)

def get_dcl(device, tau, Lambda):
    return dcl_loss(device, tau, Lambda, 0.1, 2, False, "sum", False)

def run(args, optimizer, params, sm_model, fcs, entity_emb, input_index, adj, featuress, train_ill, weight_raw, criterions, non_trains, test1, test2):
    new_links  = []
    epoch_CG   = 0
    for epoch in range(args.epochs):
        if epoch == epoch >= args.il_start:
            optimizer = optim.AdamW(params, lr=args.lr / 5)
        sm_model.train()
        for fc in fcs:
            fc.train()
        optimizer.zero_grad()

        LOSS_sum = 0
        emb0 = fcs[0](featuress[0])
        emb1 = fcs[1](featuress[1])
        emb2 = fcs[2](featuress[2])
        emb3 = sm_model(entity_emb(input_index), adj)

        epoch_CG += 1
        np.random.shuffle(train_ill)
        for si in np.arange(0, train_ill.shape[0], args.bsize):
            w_normalized = F.softmax(weight_raw, dim=0)
            EMB = torch.cat([
                w_normalized[0] * F.normalize(emb0).detach(), \
                w_normalized[1] * F.normalize(emb1).detach(), \
                w_normalized[2] * F.normalize(emb2).detach(), \
                w_normalized[3] * F.normalize(emb3).detach(), \
                ], dim=1)
            loss_ucl = criterions[0](EMB, train_ill[si:si + args.bsize])
            loss_dcl = criterions[1](EMB, train_ill[si:si + args.bsize])
            LOSS = loss_ucl * args.Lambda + loss_dcl
            LOSS.backward(retain_graph=True)
            LOSS_sum = LOSS_sum + LOSS.item()

        optimizer.step()
        if epoch >= args.il_start and (epoch+1) % args.semi_learn_step == 0 and args.il:
            with torch.no_grad():
                w_normalized = F.softmax(weight_raw, dim=0)
                EMB = torch.cat([
                    w_normalized[0] * F.normalize(fcs[0](featuress[0])), \
                    w_normalized[1] * F.normalize(fcs[1](featuress[1])), \
                    w_normalized[2] * F.normalize(fcs[2](featuress[2])), \
                    w_normalized[3] * F.normalize(sm_model(entity_emb(input_index), adj))
                ], dim=1)
                EMB = F.normalize(EMB)
            distance_list = []
            for i in np.arange(0,len(non_trains[0]), 1000):
                d = pairwise_distances(EMB[non_trains[0][i:i+1000]], EMB[non_trains[1]])
                distance_list.append(d)
            distance = torch.cat(distance_list, dim=0)
            preds1 = torch.argmin(distance, dim=1).cpu().numpy().tolist()
            preds2 = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
            del distance_list, distance, EMB

            if (epoch+1) % (args.semi_learn_step * 10) == args.semi_learn_step:
                new_links = [(non_trains[0][i],non_trains[1][p]) for i,p in enumerate(preds1) if preds2[p]==i]
            else:
                new_links = [(non_trains[0][i],non_trains[1][p]) for i,p in enumerate(preds1) if (preds2[p]==i) \
                    and ((non_trains[0][i],non_trains[1][p]) in new_links)]
        if epoch >= args.il_start and (epoch+1) % (args.semi_learn_step * 10) == 0 and len(new_links)!=0 and args.il:
            new_links_elect = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_elect)))
        for nl in new_links_elect:
            non_trains[0].remove(nl[0])
            non_trains[1].remove(nl[1])
        new_links = []

    if (epoch + 1) % args.check_point == 0:
        with torch.no_grad():
            sm_model.eval()
            fcs[0].eval()
            fcs[1].eval()
            fcs[2].eval()

            emb0 = fcs[0](featuress[0])
            emb1 = fcs[1](featuress[1])
            emb2 = fcs[2](featuress[2])
            emb3 = sm_model(entity_emb(input_index), adj)

            w_normalized = F.softmax(weight_raw, dim=0)

            EMB = torch.cat([
                    w_normalized[0] * F.normalize(emb0), \
                    w_normalized[1] * F.normalize(emb1), \
                    w_normalized[2] * F.normalize(emb2), \
                    w_normalized[3] * F.normalize(emb3), \
                ], dim=1)
            EMB = F.normalize(EMB)
            top_k = [1, 10, 50]
            if "100" in args.file_dir:
                Lvec = final_emb[test1].cpu().data.numpy()
                Rvec = final_emb[test2].cpu().data.numpy()
                acc_l2r, mean_l2r, mrr_l2r, acc_r2l, mean_r2l, mrr_r2l = multi_get_hits(Lvec, Rvec, top_k=top_k, args=args)
                del final_emb
                gc.collect()
            else:
                acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
                acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
                _, _, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
                if args.dist == 2:
                    distance = pairwise_distances(final_emb[test1], final_emb[test2])
                elif args.dist == 1:
                    distance = torch.FloatTensor(scipy.spatial.distance.cdist(\
                        final_emb[test1].cpu().data.numpy(),\
                        final_emb[test2].cpu().data.numpy(), metric="cityblock"))
                else:
                    raise NotImplementedError
                    
                if args.csls is True:
                    distance = 1 - csls_sim(1 - distance, args.csls_k)
                    
                if epoch+1 == args.epochs:
                    to_write = []
                    test_left_np = test1.cpu().numpy()
                    test_right_np = test2.cpu().numpy()
                    to_write.append(["idx","rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])
                for idx in range(test1.shape[0]):
                    values, indices = torch.sort(distance[idx, :], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_l2r += (rank + 1)
                    mrr_l2r += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_l2r[i] += 1
                    if epoch+1 == args.epochs:
                        indices = indices.cpu().numpy()
                        to_write.append([idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]], test_right_np[indices[1]], test_right_np[indices[2]]])
                if epoch+1 == args.epochs:
                    import csv
                    with open("logs/pred.txt", "w") as f:
                        wr = csv.writer(f, dialect='excel')
                        wr.writerows(to_write)

                for idx in range(test2.shape[0]):
                    _, indices = torch.sort(distance[:, idx], descending=False)
                    rank = (indices == idx).nonzero().squeeze().item()
                    mean_r2l += (rank + 1)
                    mrr_r2l += 1.0 / (rank + 1)
                    for i in range(len(top_k)):
                        if rank < top_k[i]:
                            acc_r2l[i] += 1
                mean_l2r /= test1.size(0)
                mean_r2l /= test2.size(0)
                mrr_l2r /= test1.size(0)
                mrr_r2l /= test2.size(0)
                for i in range(len(top_k)):
                    acc_l2r[i] = round(acc_l2r[i] / test1.size(0), 4)
                    acc_r2l[i] = round(acc_r2l[i] / test2.size(0), 4)
                del distance, gph_emb, img_emb, rel_emb, att_emb
                gc.collect()

if __name__ == '__main__':
    pass