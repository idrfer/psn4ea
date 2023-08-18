import gc
import math
import time
import scipy
import torch
import random
import argparse

from load import *
from utils import *
from collections import Counter

import numpy as np
import torch.nn as nn
import troch.optim as optim
import torch.nn.functional as F

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsup", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--bsize", type=int, default=5000, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--csls_k", type=int, default=10, help="top-K for CSLS")
    parser.add_argument("--unsup_k", type=int, default=1000, help="visual seed")
    parser.add_argument("--cuda", default=True, help="whether to use CUDA or not")
    parser.add_argument("--il_start", type=int, default=500, help="the start of Il")
    parser.add_argument("--dist", type=int, default=2, help="L1 distance or L2 distance")
    parser.add_argument("--rate", type=float, default=0.3, help="the rate of training set")
    parser.add_argument("--heads", type=str, default="3,3", help="heads in each GAT layer")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train")
    parser.add_argument("--csls", default=False, help="wether use CSLS for inference or not")
    parser.add_argument("--Lambda", type=float, default=0.5, help="the weight of NTXent Loss")
    parser.add_argument("--lta_split", type=int, default=0, help="split in {0,1,2,3,|split|-1}")
    parser.add_argument("--file_dir", type=str, required=True, help="input the path of datasets")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for each layers")
    parser.add_argument("--instance_normalization", default=False, help="enable instance normalization")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for each GAT layer")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay (L2 loss for parameters)")
    parser.add_argument("--check_point", type=int, default=100, help="interval between different check points")
    parser.add_argument("--tau", type=float, default=0.1, help="the temperature factor of contrastive loss")
    parser.add_argument("--hidden_units", type=str, default="200,200,100", help="hidden units in each hidden layer")    
    

    args = parser.parse_args()
    device              = run_args(args)
    split, img_vec_path = run_version(args)

    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(args.file_dir, [1,2])
    np.random.shuffle(ills)

    e1 = os.path.join(args.file_dir, "ent_ids_1")
    e2 = os.path.join(args.file_dir, "ent_ids_2")
    ents1 = get_ids(e1)
    ents2 = get_ids(e2)
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)

    img_features = load_img(ENT_NUM, img_vec_path)
    img_features = F.normalize(torch.Tensor(img_features).to(device))
    weight_raw   = torch.tensor([1.0,1.0,1.0,1.0], requires_grad=True, device=device)

    if args.unsup:
        img_f1  = img_features[ents1]
        img_f2  = img_features[ents2]
        img_sim = img_f1.mm(img_f2.t())
        topk    = args.unsup_k
        indices_2d = get_topk_indices(img_sim, topk*100)
        del img_f1, img_f2, img_sim

        visual_links = []
        used_inds    = []
        count        = 0
        for ind in indices_2d:
            if ents1[ind[0]] in used_inds: continue
            if ents2[ind[1]] in used_inds: continue
            used_inds.append(ents1[ind[0]])
            used_inds.append(ents2[ind[1]])
            visual_links.append((ents1[ind[0]], ents2[ind[1]]))
            count += 1
            if count == topk:
                break
        train_ill = np.array(visual_links, dtype=np.int32)
    else:
        train_ill = np.array(ills[:int(len(ills) // 1 * args.rate)], dtype=np.int32)
    
    test_ill_ = ills[int(len(ills)// 1 * args.rate):]
    test_ill  = np.array(test_ill_, dtype=np.int32)

    non_train1 = list(set(ents1) - set(train_ill[:, 0].tolist()))
    non_train2 = list(set(ents2) - set(train_ill[:, 1].tolist()))
    test1      = torch.LongTensor(test_ill[:, 0].squeeze()).to(device)
    test2      = torch.LongTensor(test_ill[:, 1].squeeze()).to(device)
    a1 = os.path.join(args.file_dir, "training_attrs_1")
    a2 = os.path.join(args.file_dir, "training_attrs_2")

    att_features = load_attr([a1, a2], ENT_NUM, ent2id_dict, 1000)
    att_features = torch.Tensor(att_features).to(device)
    rel_features = load_relation(ENT_NUM, triples, 1000)
    rel_features = torch.Tensor(rel_features).to(device)

    rel_fc = nn.Linear(1000, 100).to(device)
    att_fc = nn.Linear(1000, 100).to(device)
    img_fc = nn.Linear(img_features.shape[1], 200).to(device)

    input_dim = int(args.hidden_units.strip().split(",")[0])
    entity_emb = nn.Embedding(ENT_NUM, input_dim)
    nn.init.normal_(entity_emb.weight, std=1.0 / math.sqrt(ENT_NUM))
    entity_emb.requires_grad = True
    entity_emb = entity_emb.to(device)

    input_idx = torch.LongTensor(np.arange(ENT_NUM)).to(device)
    adj = get_adjr(ENT_NUM, triples, norm=True)
    adj = adj.to(device)

    sm_model = get_model("GAT", args, device)

    for param in img_fc.parameters():
        param.requires_grad = True
    params = [
            {"params":
            list(sm_model.parameters())+
            list(img_fc.parameters()) + 
            list(rel_fc.parameters()) + 
            list(att_fc.parameters()) +
            [entity_emb.weight] + 
            [weight_raw]
        }]
    optimizer = optim.AdamW(
            params,
            lr=args.lr
    )

    criterion_ucl = get_ucl(device, args.tau, args.Lambda, 2)
    criterion_dcl = get_dcl(device, args.tau, args.Lambda)

    run(args, optimizer, params, sm_model, [img_fc, rel_fc, att_fc], entity_emb, input_idx, adj, [img_features, rel_features, att_features], train_ill, weight_raw, [criterion_ucl, criterion_dcl], [non_train1, non_train2], test1, test2)

    pass

if __name__ == "__main__":
    main()
