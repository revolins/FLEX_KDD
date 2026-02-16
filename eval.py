
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from torch_geometric.utils import negative_sampling
import os
import torch_geometric.transforms as T
from tqdm import tqdm

def evaluate_hits(evaluator, pos_pred, neg_pred, k_list):
    results = {}
    for K in k_list:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']

        hits = round(hits, 4)
        results[f'Hits@{K}'] = hits

    return results

def get_hit_score(pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, neg_train_pred):
    
    k_list = [1, 3, 10, 20, 50, 100]
    result = {}
    evaluator_hit = Evaluator(name='ogbl-collab')

    result_mrr_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_train_pred, k_list)
    result_mrr_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_mrr_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)
    
    #result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'Hits@{K}'], result_mrr_val[f'Hits@{K}'], result_mrr_test[f'Hits@{K}'])

    return result

def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    # test_auc = roc_auc_score(test_true, test_pred)
    results = {}
    
    valid_auc = round(valid_auc, 4)
    # test_auc = round(test_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    # test_ap = average_precision_score(test_true, test_pred)
    
    valid_ap = round(valid_ap, 4)
    # test_ap = round(test_ap, 4)
    
    results['AP'] = valid_ap


    return results 

def evaluate_mrr(pos_val_pred, neg_val_pred):
    results = {}

    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    mrr_output =  eval_mrr(pos_val_pred, neg_val_pred)

    valid_mrr = mrr_output['mrr_list'].mean().item()
    valid_mrr_hit1 = mrr_output['hits@1_list'].mean().item()
    valid_mrr_hit3 = mrr_output['hits@3_list'].mean().item()
    valid_mrr_hit10 = mrr_output['hits@10_list'].mean().item()
    valid_mrr_hit20 = mrr_output['hits@20_list'].mean().item()
    valid_mrr_hit50 = mrr_output['hits@50_list'].mean().item()
    valid_mrr_hit100 = mrr_output['hits@100_list'].mean().item()

    valid_mrr = round(valid_mrr, 4)
    valid_mrr_hit1 = round(valid_mrr_hit1, 4)
    valid_mrr_hit3 = round(valid_mrr_hit3, 4)
    valid_mrr_hit10 = round(valid_mrr_hit10, 4)
    valid_mrr_hit20 = round(valid_mrr_hit20, 4)
    valid_mrr_hit50 = round(valid_mrr_hit50, 4)
    valid_mrr_hit100 = round(valid_mrr_hit100, 4)
    
    results['MRR'] = valid_mrr
    results['mrr_hit1'] = valid_mrr_hit1
    results['mrr_hit3'] = valid_mrr_hit3
    results['mrr_hit10'] = valid_mrr_hit10
    results['mrr_hit20'] = valid_mrr_hit20
    results['mrr_hit50'] = valid_mrr_hit50
    results['mrr_hit100'] = valid_mrr_hit100

    return results

def eval_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''
    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)

    return { 'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@20_list': hits20_list,
                'hits@50_list': hits50_list,
                'hits@10_list': hits10_list,
                'hits@100_list': hits100_list,
                'mrr_list': mrr_list}

def get_metric_score_orig(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])

    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])

    
    return result

def get_metric_score(pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, neg_train_pred):
    
    k_list = [1, 3, 10, 20, 50, 100]
    result = {}

    result_mrr_train = evaluate_mrr(pos_train_pred, neg_train_pred)
    result_mrr_val = evaluate_mrr(pos_val_pred, neg_val_pred)
    result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred)
    
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result

@torch.no_grad()
def test_hardedge(score_func, input_data, h, batch_size,  negative_data=None):

    pos_preds = []
    neg_preds = []

    if negative_data is not None:
        
        for perm in tqdm(DataLoader(range(input_data.size(0)),  batch_size)):
            #print(perm)
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))

            #print("pos_edges.size: ", pos_edges.size())
            #print("h size: ", h.size(), flush=True)
            pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
            neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()

            pos_preds += [pos_scores]
            neg_preds += [neg_scores]
        
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm in tqdm(DataLoader(range(input_data.size(0)), batch_size)):
            edge = input_data[perm].t()
            pos_preds += [score_func(x_i=h[edge[0]], x_j=h[edge[1]]).cpu()]
            
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds, neg_preds

@torch.no_grad()
def test_hard(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, tr_batch_size, v_batch_size, batch_size, ood_method):
    model.eval()
    score_func.eval()

    pos_train_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge, neg_train_edge = evaluation_edges

    if emb == None: x = data.x
    else: x = emb.weight
    
    h = model(x, data.adj_t.to(x.device))
    x1 = h
    x2 = torch.tensor(1)
 
    pos_train_edge = pos_train_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)
    neg_train_edge = neg_train_edge.to(x.device)

    pos_valid_pred, neg_valid_pred = test_hardedge(score_func, pos_valid_edge, h, v_batch_size, negative_data=neg_valid_edge)
    pos_test_pred, neg_test_pred = test_hardedge(score_func, pos_test_edge, h, batch_size, negative_data=neg_test_edge)
    if neg_train_edge.dim() == 2: neg_train_edge = neg_train_edge.unsqueeze(1)
    pos_train_pred, neg_train_pred = test_hardedge(score_func, pos_train_edge, h, tr_batch_size, negative_data=neg_train_edge)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_train_pred = neg_train_pred.squeeze(-1)
   
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)
    
    print("neg_train_pred size before predictions: ", neg_train_pred.size(), flush=True)
    neg_train_pred = neg_train_pred[:, 0:1] # DEBUG -- reduce torch.Size() to 1
   
    print('train_pos train_neg valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), neg_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    result = get_metric_score(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred)
    score_emb = []

    return result, score_emb

@torch.no_grad()
def test_edge(model, score_func, loader, device, args):

    y_pred, y_true = [], []
    for data in tqdm(loader, ncols=70):
        data = data.to(device)
        
        h = model(data.x, data.adj_t, data.z)
        edge = data.edge.t()

        logits = score_func(x_i=h[edge[0]], x_j=h[edge[1]])
        
        y_pred.append(logits.view(-1))
        y_true.append(data.y.view(-1).to(torch.float))
    g_pred, g_true = torch.cat(y_pred), torch.cat(y_true)

    pos_pred = g_pred[g_true==1]
    neg_pred = g_pred[g_true==0]

    return pos_pred, neg_pred


@torch.no_grad()
def test(model, score_func, data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, batch_size, use_valedges_as_input, args):
    model.eval()
    score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge, _ = evaluation_edges

    if emb == None: x = data.x
    else: x = emb.weight
    
    h, _ = model(x, data.adj_t.to(x.device))
   
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)


    neg_valid_pred = test_edge(score_func, neg_valid_edge, h, batch_size)

    pos_valid_pred = test_edge(score_func, pos_valid_edge, h, batch_size)

    if use_valedges_as_input: h = model(x, data.full_adj_t.to(x.device))

    pos_test_pred = test_edge(score_func, pos_test_edge, h, batch_size)

    neg_test_pred = test_edge(score_func, neg_test_edge, h, batch_size)

    pos_train_pred = test_edge(score_func, train_val_edge, h, batch_size)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)

    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score_orig(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)

    return result, []

@torch.no_grad()
def gnn_test(model, score_func, train_loader, valid_loader, test_loader, device, args):
    model.eval()
    score_func.eval()
    
    pos_valid_pred, neg_valid_pred = test_edge(model, score_func, valid_loader, device, args)
    pos_test_pred, neg_test_pred = test_edge(model, score_func, test_loader, device, args)
    pos_train_pred, neg_train_pred = test_edge(model, score_func, train_loader, device, args)

    pos_train_pred = torch.flatten(pos_train_pred)
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)

    if not args.rand_samp:
        neg_train_pred = neg_train_pred.view(pos_train_pred.size(0), -1)
        neg_valid_pred = neg_valid_pred.view(pos_valid_pred.size(0), -1)
        neg_test_pred = neg_test_pred.view(pos_test_pred.size(0), -1)
        
        return get_metric_score(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred)
    
    neg_train_pred = torch.flatten(neg_train_pred)
    neg_valid_pred = torch.flatten(neg_valid_pred)
    neg_test_pred = torch.flatten(neg_test_pred)

    print(f"pos_train: {pos_train_pred.size()}, neg_train: {neg_train_pred.size()}, pos_val: {pos_valid_pred.size()}, neg_val: {neg_valid_pred.size()}, pos_test: {pos_test_pred.size()}, neg_test: {neg_test_pred.size()}")
    
    return get_hit_score(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred)

class PermIterator:

    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret

def ncn_train(model,
          predictor,
          data,
          split_edge,
          optimizer,
          batch_size,
          maskinput: bool = True,
          cnprobs: Iterable[float]=[],
          alpha: float=None):
    def penalty(posout, negout):
        scale = torch.ones_like(posout[[0]]).requires_grad_()
        loss = -F.logsigmoid(posout*scale).mean()-F.logsigmoid(-negout*scale).mean()
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(torch.square(grad))
    
    # if alpha is not None:
    #     predictor.setalpha(alpha)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)
    pos_train_edge = pos_train_edge.t()


    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm in PermIterator(
            adjmask.device, adjmask.shape[0], batch_size
    ):
        
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = model(data.x, adj)
        
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge,
                                                    cndropprobs=cnprobs)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge, cndropprobs=cnprobs)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss

@torch.no_grad()
def ncn_test(model, predictor, data, split_edge, evaluator, batch_size,
         use_valedges_as_input):
    model.eval()
    predictor.eval()

    #pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)

    # pos_train_pred = torch.cat([
    #     predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
    #     for perm in PermIterator(pos_train_edge.device,
    #                              pos_train_edge.shape[0], batch_size, False)
    # ],
    #                            dim=0)
    

    pos_valid_pred = torch.cat([
        predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    neg_valid_pred = torch.cat([
        predictor(h, adj, neg_valid_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_valid_edge.device,
                                 neg_valid_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)

    pos_test_pred = torch.cat([
        predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    neg_test_pred = torch.cat([
        predictor(h, adj, neg_test_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(neg_test_edge.device, neg_test_edge.shape[0],
                                 batch_size, False)
    ],
                              dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K

        train_hits = 0
        # evaluator.eval({
        #     'y_pred_pos': pos_train_pred,
        #     'y_pred_neg': neg_valid_pred,
        # })[f'hits@{K}']


        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
    

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)
    # print(predictor.calilin.weight, predictor.calilin.bias)
    # print(predictor.pt)

    train_auc = 0
    val_pred = torch.cat([pos_valid_pred, neg_valid_pred])
    val_true = torch.cat([torch.ones(pos_valid_pred.size(0), dtype=int), 
                            torch.zeros(neg_valid_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])
    
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    results['AUC'] = (train_auc, result_auc_val['AUC'], result_auc_test['AUC'])
    results['AP'] = (train_auc, result_auc_val['AP'], result_auc_test['AP'])

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(),  h.cpu()]


    return results, score_emb

@torch.no_grad()
def ncn_hard_test(model, predictor, data, split_edge, evaluator_mrr, evaluator_hit, batch_size,
         use_valedges_as_input):
    model.eval()
    predictor.eval()

    # pos_train_edge = split_edge['train']['edge'].to(data.adj_t.device())
    pos_valid_edge = split_edge['valid']['edge'].to(data.adj_t.device())
    neg_valid_edge = split_edge['valid']['edge_neg'].to(data.adj_t.device())
    pos_test_edge = split_edge['test']['edge'].to(data.adj_t.device())
    neg_test_edge = split_edge['test']['edge_neg'].to(data.adj_t.device())

    adj = data.adj_t
    h = model(data.x, adj)
    neg_num = neg_test_edge.size(1)

    '''
    pos_train_pred = torch.cat([
        predictor(h, adj, pos_train_edge[perm].t()).squeeze().cpu()
        for perm in PermIterator(pos_train_edge.device,
                                 pos_train_edge.shape[0], batch_size, False)
    ],
                               dim=0)
    '''
    pos_preds = []
    neg_preds = []

    for perm in PermIterator(pos_valid_edge.device,
                                 pos_valid_edge.shape[0], batch_size, False):
        
        pos_preds += [predictor(h, adj, pos_valid_edge[perm].t()).squeeze().cpu()]
        
        neg_edges = torch.permute(neg_valid_edge[perm], (2, 0, 1))
        neg_edges = neg_edges.view(2,-1)
        neg_preds += [predictor(h, adj, neg_edges).squeeze().cpu()]
    
    pos_valid_pred = torch.cat(pos_preds, dim=0)
    neg_valid_pred = torch.cat(neg_preds, dim=0)
    
    if use_valedges_as_input:
        adj = data.full_adj_t
        h = model(data.x, adj)

    pos_preds = []
    neg_preds = []
    for perm in PermIterator(pos_test_edge.device, pos_test_edge.shape[0],
                                 batch_size, False):
        pos_preds += [predictor(h, adj, pos_test_edge[perm].t()).squeeze().cpu()] 
        
        neg_edges = torch.permute(neg_test_edge[perm], (2, 0, 1))
        neg_edges = neg_edges.view(2,-1)
        neg_preds += [predictor(h, adj, neg_edges).squeeze().cpu()]
    
    pos_test_pred = torch.cat(pos_preds, dim=0)
    neg_test_pred = torch.cat(neg_preds, dim=0)
    
    neg_valid_pred = neg_valid_pred.view(-1, neg_num)
    neg_test_pred = neg_test_pred.view(-1, neg_num)

    
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)
    pos_train_pred = pos_valid_pred

    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, pos_train_pred)

    return result, []