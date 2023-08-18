import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network_com import *

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)
        return x
    
class GCN(nn.Module):
    def __init__(self, n_units, dropout):
        super(GCN, self).__init__()
        self.nfeat = n_units[0]
        self.nhid  = n_units[1]
        self.nout  = n_units[2]
        
        self.gc1 = GraphConvolution(self.nfeat, self.nhid)
        self.gc2 = GraphConvolution(self.nhid,  selfnout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

def cosine_sim(im, s):
    return im.mm(s.t())

def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X) + 1e-8
    X = torch.div(X, a)    
    return X

class NCA_loss(nn.Module):

    def __init__(self, alpha, beta, ep):
        super(NCA_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.sim = cosine_sim

    def forward(self, emb, train_links, test_links, device=0):
        
        emb = F.normalize(emb)

        im = emb[train_links[:, 0]]
        s = emb[train_links[:,1]]
    
        
        if len(test_links) != 0:
            test_links = test_links[random.sample([x for x in np.arange(0,len(test_links))],4500)]

            im_neg_scores = self.sim(im, emb[test_links[:,1]])
            s_neg_scores = self.sim(s, emb[test_links[:,0]])
        
        bsize = im.size()[0]
        scores = self.sim(im, s) #+ 1
        tmp  = torch.eye(bsize).cuda(device)
        s_diag = tmp * scores
        
        alpha = self.alpha
        alpha_2 = alpha
        beta = self.beta
        ep = self.ep
        S_ = torch.exp(alpha * (scores - ep))
        S_ = S_ - S_ * tmp

        if len(test_links) != 0:
            S_1 = torch.exp(alpha * (im_neg_scores - ep))
            S_2 = torch.exp(alpha * (s_neg_scores - ep))

        loss_diag = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum(
                torch.log(1 + S_.sum(0)) / alpha
                + torch.log(1 + S_.sum(1)) / alpha 
                + loss_diag * beta \
                ) / bsize
        if len(test_links) != 0:
            loss_global_neg = (torch.sum(torch.log(1 + S_1.sum(0)) / alpha_2
                + torch.log(1 + S_2.sum(0)) / alpha_2) 
                + torch.sum(torch.log(1 + S_1.sum(1)) / alpha_2
                + torch.log(1 + S_2.sum(1)) / alpha_2)) / 4500 
        if len(test_links) != 0:
            return loss + loss_global_neg
        return loss
    
class NCA_loss_cross_modal(nn.Module):

    def __init__(self, alpha, beta, ep):
        super(NCA_loss_cross_modal, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ep = ep
        self.sim = cosine_sim

    def forward(self, emb1, emb2, train_links, device=0):
        
        emb1 = F.normalize(emb1)
        emb2 = F.normalize(emb2)

        im = emb1[train_links[:, 0]]
        s = emb2[train_links[:,1]]

        bsize = im.size()[0]
        scores = self.sim(im, s)
        tmp  = torch.eye(bsize).cuda(device)
        s_diag = tmp * scores
        
        alpha = self.alpha
        alpha_2 = alpha
        beta = self.beta
        ep = self.ep
        S_ = torch.exp(alpha * (scores - ep))
        S_ = S_ - S_ * tmp

        loss_diag = - torch.log(1 + F.relu(s_diag.sum(0)))

        loss = torch.sum(
                torch.log(1 + S_.sum(0)) / alpha
                + torch.log(1 + S_.sum(1)) / alpha 
                + loss_diag * beta \
                ) / bsize
        return loss

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        nn.init.xavier_uniform(self.att_weights.data)

    def get_mask(self):
        pass

    def forward(self, inputs):

        if self.batch_first:
            batch_size, _ = inputs.size()[:2]
        else:
            _, batch_size = inputs.size()[:2]
            inputs        = inputs.permute(1, 0, 2)

        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        attentions = F.softmax(F.relu(weights.squeeze()))
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))
        representations = weighted.sum(1).squeeze()
        return representations, attentions


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size  = d_model
        self.eps   = eps
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias  = nn.Parameter(torch.zeros(self.size))
        
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = concat
        output = self.out(concat)
    
        return output

class ucl_loss(nn.Module):

    def __init__(self, device, tau=0.05, ab_weight=0.5, n_view=2, intra_weight=1.0, inversion=False):
        super(ucl_loss, self).__init__()
        self.tau = tau
        self.device = device
        self.sim = cosine_sim
        self.weight = ab_weight 
        self.n_view = n_view
        self.intra_weight = intra_weight
        self.inversion = inversion

    def softXEnt(self, target, logits):

        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, emb, train_links, emb2=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
            if emb2 is not None:
                emb2 = F.normalize(emb2, dim=1)
        zis = emb[train_links[:, 0]]
        if emb2 is not None:
            zjs = emb2[train_links[:, 1]]
        else:
            zjs = emb[train_links[:, 1]]

        temp = self.tau
        alpha = self.weight
        n_view = self.n_view

        LARGE_NUM = 1e9

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        num_classes = batch_size * n_view
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
        labels = labels.to(self.device)

        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.to(self.device).float()
        logits_a2a = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temp
        logits_a2a = logits_a2a - masks * LARGE_NUM
        logits_b2b = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temp
        logits_b2b = logits_b2b - masks * LARGE_NUM
        logits_a2b = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temp
        logits_b2a = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temp

        if self.inversion:
            logits_a = torch.cat([logits_a2b, logits_b2b], dim=1)
            logits_b = torch.cat([logits_b2a, logits_a2a], dim=1)
        else:
            logits_a = torch.cat([logits_a2b, logits_a2a], dim=1)
            logits_b = torch.cat([logits_b2a, logits_b2b], dim=1)

        loss_a = self.softXEnt(labels, logits_a)
        loss_b = self.softXEnt(labels, logits_b)

        return alpha * loss_a + (1 - alpha) * loss_b
    
class dcl_loss(nn.Module):

    def __init__(self, device, tau=0.05, ab_weight=0.5, zoom=0.1, n_view=2, inversion=False, reduction="mean", detach=False):
        super(dcl_loss, self).__init__()
        self.tau = tau
        self.device = device
        self.sim = cosine_sim
        self.weight = ab_weight
        self.zoom = zoom
        self.n_view = n_view
        self.inversion = inversion
        self.reduction = reduction
        self.detach = detach

    def forward(self, emb1, emb2, train_links, norm=True):
        if norm:
            emb1 = F.normalize(emb1, dim=1)
            emb2 = F.normalize(emb2, dim=1)

        src_zis = emb1[train_links[:, 0]]
        src_zjs = emb1[train_links[:, 1]]
        tar_zis = emb2[train_links[:, 0]]
        tar_zjs = emb2[train_links[:, 1]]

        temp = self.tau
        alpha = self.weight

        assert src_zis.shape[0] == tar_zjs.shape[0]
        batch_size = src_zis.shape[0]
        LARGE_NUM = 1e9
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.to(self.device).float()
        p_a2b = torch.matmul(src_zis, torch.transpose(src_zjs, 0, 1)) / temp
        p_b2a = torch.matmul(src_zjs, torch.transpose(src_zis, 0, 1)) / temp
        q_a2b = torch.matmul(tar_zis, torch.transpose(tar_zjs, 0, 1)) / temp
        q_b2a = torch.matmul(tar_zjs, torch.transpose(tar_zis, 0, 1)) / temp
        p_a2a = torch.matmul(src_zis, torch.transpose(src_zis, 0, 1)) / temp
        p_b2b = torch.matmul(src_zjs, torch.transpose(src_zjs, 0, 1)) / temp
        q_a2a = torch.matmul(tar_zis, torch.transpose(tar_zis, 0, 1)) / temp
        q_b2b = torch.matmul(tar_zjs, torch.transpose(tar_zjs, 0, 1)) / temp
        p_a2a = p_a2a - masks * LARGE_NUM
        p_b2b = p_b2b - masks * LARGE_NUM
        q_a2a = q_a2a - masks * LARGE_NUM
        q_b2b = q_b2b - masks * LARGE_NUM

        if self.inversion:
            p_a2b = torch.cat([p_a2b, p_b2b], dim=1)
            p_b2a = torch.cat([p_b2a, p_a2a], dim=1)
            q_a2b = torch.cat([q_a2b, q_b2b], dim=1)
            q_b2a = torch.cat([q_b2a, q_a2a], dim=1)
        else:
            p_a2b = torch.cat([p_a2b, p_a2a], dim=1)
            p_b2a = torch.cat([p_b2a, p_b2b], dim=1)
            q_a2b = torch.cat([q_a2b, q_a2a], dim=1)
            q_b2a = torch.cat([q_b2a, q_b2b], dim=1)

        loss_a = F.kl_div(F.log_softmax(p_a2b, dim=1), F.softmax(q_a2b.detach(), dim=1), reduction="none")
        loss_b = F.kl_div(F.log_softmax(p_b2a, dim=1), F.softmax(q_b2a.detach(), dim=1), reduction="none")

        if self.reduction == "mean":
            loss_a = loss_a.mean()
            loss_b = loss_b.mean()
        elif self.reduction == "sum":
            loss_a = loss_a.sum()
            loss_b = loss_b.sum()
        return self.zoom * (alpha * loss_a + (1 - alpha) * loss_b)
