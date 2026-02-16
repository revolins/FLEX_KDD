import torch
import numpy as np
import random
import logging, sys
import math
import logging.config 
import networkx as nx
import scipy.sparse as ssp
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata, ks_2samp
from torch_sparse import SparseTensor
from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (add_self_loops, negative_sampling, degree, to_undirected, train_test_split_edges, to_edge_index, is_sparse)
from collections import Counter

##SIG-VAE Utils

def score_sparse(recovered, data, pos_scores, neg_scores, pos_num_nodes, neg_num_nodes):

    #flex_ei = recovered.nonzero().t().contiguous().cpu()
    if not is_sparse(recovered): recovered = recovered.to_sparse()
    flex_ei, flex_values = to_edge_index(recovered)
    flex_ei = flex_ei.cpu()
    _, edge_ids = torch.unique(data.edge.t(), return_inverse=True)
    
    A = csr_matrix((torch.ones(flex_ei.size(1), dtype=int), (flex_ei[0], flex_ei[1])), 
                shape=(recovered.size(0), recovered.size(1)))

    rows, cols = A.nonzero()
    A[cols, rows] = A[rows, cols]
    
    src, dst = edge_ids[0].cpu(), edge_ids[1].cpu()
    cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()

    if len(data.y) == 1 and data.y == 1:
        pos_scores.extend(cur_scores)
    elif len(data.y) == 1 and data.y == 0:
        neg_scores.extend(cur_scores)
    elif len(data.y) > 1:
        pos_scores.extend(cur_scores[data.y.view(-1).cpu().numpy() == 1])
        neg_scores.extend(cur_scores[data.y.view(-1).cpu().numpy() == 0])
    else:
        raise AssertionError(f"Data during VGAE Generation on Run #{t} is unlabelled")
    
    pos_num_nodes.extend(np.array(list(Counter(data.batch.cpu().numpy()).values()))[data.y.view(-1).cpu().numpy() == 1])
    neg_num_nodes.extend(np.array(list(Counter(data.batch.cpu().numpy()).values()))[data.y.view(-1).cpu().numpy() == 0])
    
    return pos_scores, neg_scores, pos_num_nodes, neg_num_nodes

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = ssp.coo_matrix(adj)
    adj_ = adj + ssp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = ssp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def format_scores(split_edge, A, score_log, args):

    train_scores, _ = CN(A, split_edge['train']['edge'].t())
    train_n_scores, _ = CN(A, split_edge['train']['edge_neg'].transpose(1, 0))
    valid_scores, _ = CN(A, split_edge['valid']['edge'].t())
    test_scores, _ = CN(A, split_edge['test']['edge'].t())
    valid_n_scores, _ = CN(A, torch.permute(split_edge['valid']['edge_neg'], (2, 0, 1)).view(2, -1))
    test_n_scores, _ = CN(A, torch.permute(split_edge['test']['edge_neg'], (2, 0, 1)).view(2, -1))

    train_scores = np.array(train_scores)
    train_n_scores = np.array(train_n_scores)
    valid_scores = np.array(valid_scores)
    test_scores = np.array(test_scores)
    valid_n_scores = np.array(valid_n_scores)
    test_n_scores = np.array(test_n_scores)

    plot_test = np.append(test_scores, np.random.normal(loc=np.mean(test_scores), scale=1, size=len(test_scores) // 1000))
    plot_neg_test = np.append(test_n_scores, np.random.normal(loc=np.mean(test_n_scores), scale=1, size=len(test_n_scores) // 1000))
    plot_val = np.append(valid_scores, np.random.normal(loc=np.mean(valid_scores), scale=1, size=len(valid_scores) // 1000))
    plot_neg_val = np.append(valid_n_scores, np.random.normal(loc=np.mean(valid_n_scores), scale=1, size=len(valid_n_scores) // 1000))
    plot_train = np.append(train_scores, np.random.normal(loc=np.mean(train_scores), scale=1, size=len(train_scores)// 1000))
    plot_neg_train = np.append(train_n_scores, np.random.normal(loc=np.mean(train_n_scores), scale=1, size=len(train_n_scores)// 1000))

    with open(score_log, "a") as file:
        file.write(f"Initial Args: {args}\n")
        file.write(f"Positive Initial Train: {np.mean(train_scores)} ± {np.var(train_scores)}, Median: {np.median(train_scores)}, 25% q: {np.quantile(train_scores, 0.25)}, 50% q: {np.quantile(train_scores, 0.5)}, 75% q: {np.quantile(train_scores, 0.75)}, 90% q: {np.quantile(train_scores, 0.9)}\n")
        file.write(f"Negative Initial Train: {np.mean(train_n_scores)} ± {np.var(train_n_scores)}, Median: {np.median(train_n_scores)}, 25% q: {np.quantile(train_n_scores, 0.25)}, 50% q: {np.quantile(train_n_scores, 0.5)}, 75% q: {np.quantile(train_n_scores, 0.75)}, 90% q: {np.quantile(train_n_scores, 0.9)}\n")
        file.write(f"Positive Initial Valid: {np.mean(valid_scores)} ± {np.var(valid_scores)}, Median: {np.median(valid_scores)}, 25% q: {np.quantile(valid_scores, 0.25)}, 50% q: {np.quantile(valid_scores, 0.5)}, 75% q: {np.quantile(valid_scores, 0.75)}, 90% q: {np.quantile(valid_scores, 0.9)}\n")
        file.write(f"Negative Initial Valid: {np.mean(valid_n_scores)} ± {np.var(valid_n_scores)}, Median: {np.median(valid_n_scores)}, 25% q: {np.quantile(valid_n_scores, 0.25)}, 50% q: {np.quantile(valid_n_scores, 0.5)}, 75% q: {np.quantile(valid_n_scores, 0.75)}, 90% q: {np.quantile(valid_n_scores, 0.9)}\n")
        file.write(f"Positive Initial Test: {np.mean(test_scores)} ± {np.var(test_scores)}, Median: {np.median(test_scores)}, 25% q: {np.quantile(test_scores, 0.25)}, 50% q: {np.quantile(test_scores, 0.5)}, 75% q: {np.quantile(test_scores, 0.75)}, 90% q: {np.quantile(test_scores, 0.9)}\n")
        file.write(f"Negative Initial Test: {np.mean(test_n_scores)} ± {np.var(test_n_scores)}, Median: {np.median(test_n_scores)}, 25% q: {np.quantile(test_n_scores, 0.25)}, 50% q: {np.quantile(test_n_scores, 0.5)}, 75% q: {np.quantile(test_n_scores, 0.75)}, 90% q: {np.quantile(test_n_scores, 0.9)}\n")

    return (plot_train, plot_neg_train, plot_val, plot_neg_val, plot_test, plot_neg_test)

def plot_init_score_dist(initial_scores, args, test_path):
    plot_train, plot_neg_train, plot_val, plot_neg_val, plot_test, plot_neg_test = initial_scores

    fill_b = False

    sns.kdeplot(plot_train, label=f'Pos. Train', fill=fill_b, color='orange', alpha=0.5, bw_adjust=1)
    sns.kdeplot(plot_neg_train, label=f'Neg. Train', fill=fill_b, color='orange', alpha=0.5, bw_adjust=1)
    sns.kdeplot(plot_val, label=f'Pos. Valid', fill=fill_b, color='red', alpha=0.5, bw_adjust=1)
    sns.kdeplot(plot_neg_val, label=f'Neg. Valid', fill=fill_b, color='red', alpha=0.5, bw_adjust=1)
    sns.kdeplot(plot_test, label=f'Pos. Test', fill=fill_b, color='blue', alpha=0.5, bw_adjust=1)
    sns.kdeplot(plot_neg_test, label=f'Neg. Test', fill=fill_b, color='blue', alpha=0.5, bw_adjust=1)

    plt.title(f'Flex Original {args.dataset} - CN', fontsize=18)
    plt.ylabel('Density', fontsize=16)
    plt.xlabel('Score', fontsize=16)
    plt.xlim(left=-1, right=6)
    plt.legend(fontsize='x-large')
    plt.tight_layout()
    plt.savefig(test_path + f'initial_plot.png', dpi=300)
    plt.clf()

def plot_heur_score_dist(initial_scores, pos_scores, neg_scores, pos_num_nodes, neg_num_nodes, args, epoch, test_path):
    fill_b = False
    plot_train, plot_neg_train, plot_val, plot_neg_val, plot_test, plot_neg_test = initial_scores

    pos_scores, neg_scores, pos_num_nodes, neg_num_nodes = np.array(pos_scores), np.array(neg_scores), np.array(pos_num_nodes), np.array(neg_num_nodes)

    # pos_scores = np.concatenate([ps for ps in pos_scores if ps.size > 0], axis=0)
    # neg_scores = np.concatenate([ns for ns in neg_scores if ns.size > 0], axis=0)
    
    #pos_scores = np.append(pos_scores, np.random.normal(loc=np.mean(pos_scores), scale=1, size=len(pos_scores)// 1000))
    #neg_scores = np.append(neg_scores, np.random.normal(loc=np.mean(neg_scores), scale=1, size=len(neg_scores)// 1000))

    with open(test_path + "score.txt", "a") as file:
        if epoch == 0:
            file.write(f"KS 2-samp test Initial Pos. Train vs. Pos. Scores - {epoch}: {ks_2samp(plot_train, pos_scores).pvalue}\n")
            file.write(f"KS 2-samp test Initial Neg. Train vs. Neg. Scores - {epoch}: {ks_2samp(plot_neg_train, neg_scores).pvalue}\n")
            file.write(f"KS 2-samp test - Pos Initial CN Scores vs Num Nodes: {ks_2samp(pos_num_nodes, pos_scores).pvalue}\n")
            file.write(f"KS 2-samp test - Neg Initial CN Scores vs Num Nodes: {ks_2samp(neg_num_nodes, neg_scores).pvalue}\n")
            file.write(f"Pos NN vs CN: {ks_2samp(pos_num_nodes, pos_scores).pvalue}\n")
            file.write(f"Neg NN vs CN: {ks_2samp(neg_num_nodes, neg_scores).pvalue}\n")
            file.write(f"Pos NN vs Neg CN: {ks_2samp(pos_num_nodes, neg_scores).pvalue}\n")
            file.write(f"Neg NN vs Pos CN: {ks_2samp(neg_num_nodes, pos_scores).pvalue}\n")

            file.write(f"Initial CN Pos Scores: {np.mean(plot_train) } ± {np.var(plot_train)}, Median: {np.median(plot_train)}, 25% q: {np.quantile(plot_train, 0.25)}, 50% q: {np.quantile(plot_train, 0.5)}, 75% q: {np.quantile(plot_train, 0.75)}, 90% q: {np.quantile(plot_train, 0.9)}\n")
            file.write(f"Initial CN Neg Scores: {np.mean(plot_neg_train) } ± {np.var(plot_neg_train)}, Median: {np.median(plot_neg_train)}, 25% q: {np.quantile(plot_neg_train, 0.25)}, 50% q: {np.quantile(plot_neg_train, 0.5)}, 75% q: {np.quantile(plot_neg_train, 0.75)}, 90% q: {np.quantile(plot_neg_train, 0.9)}\n")
            file.write(f"Positive Num Nodes Input: {np.mean(pos_num_nodes) } ± {np.var(pos_num_nodes)}, Median: {np.median(pos_num_nodes)}, 25% q: {np.quantile(pos_num_nodes, 0.25)}, 50% q: {np.quantile(pos_num_nodes, 0.5)}, 75% q: {np.quantile(pos_num_nodes, 0.75)}, 90% q: {np.quantile(pos_num_nodes, 0.9)}\n")
            file.write(f"Negative Num Nodes Input: {np.mean(neg_num_nodes) } ± {np.var(neg_num_nodes)}, Median: {np.median(neg_num_nodes)}, 25% q: {np.quantile(neg_num_nodes, 0.25)}, 50% q: {np.quantile(neg_num_nodes, 0.5)}, 75% q: {np.quantile(neg_num_nodes, 0.75)}, 90% q: {np.quantile(neg_num_nodes, 0.9)}\n")
        file.write(f"Epoch: {epoch} - Positive Flex Train: {np.mean(pos_scores)} ± {np.var(pos_scores)}, Median: {np.median(pos_scores)}, 25% q: {np.quantile(pos_scores, 0.25)}, 50% q: {np.quantile(pos_scores, 0.5)}, 75% q: {np.quantile(pos_scores, 0.75)}, 90% q: {np.quantile(pos_scores, 0.9)}\n")
        file.write(f"Epoch: {epoch} - Negative Flex Train: {np.mean(neg_scores) } ± {np.var(neg_scores)}, Median: {np.median(neg_scores)}, 25% q: {np.quantile(neg_scores, 0.25)}, 50% q: {np.quantile(neg_scores, 0.5)}, 75% q: {np.quantile(neg_scores, 0.75)}, 90% q: {np.quantile(neg_scores, 0.9)}\n")
        file.write("---" * 20 + "\n")

    if epoch % args.plot_interval == 0:
        sns.kdeplot(pos_scores, label=f'Pos. Train -- Epoch {epoch + 1}', fill=fill_b, color='black', alpha=1, bw_adjust=1)
        sns.kdeplot(neg_scores, label=f'Neg. Train -- Epoch {epoch + 1}', fill=fill_b, color='black', linestyle='--', alpha=1, bw_adjust=1)

        plt.title(f'Flex Generated {args.dataset} - Epoch #{epoch+1} - CN', fontsize=18)
        plt.ylabel('Density', fontsize=16)
        plt.xlabel('Score', fontsize=16)
        plt.xlim(left=-1)
        plt.legend(fontsize='x-large')
        plt.tight_layout()
        plt.savefig(test_path + f'Epoch{epoch+1}_plot.png', dpi=300)
        plt.clf()

        sns.kdeplot(pos_scores, label=f'Positive CNs', fill=fill_b, color='black', alpha=1, bw_adjust=1)
        sns.kdeplot(neg_scores, label=f'Negative CNs', fill=fill_b, color='black', linestyle='--', alpha=1, bw_adjust=1)
        sns.kdeplot(pos_num_nodes, label=f'Positive Num Nodes', fill=fill_b, color='blue', alpha=1, bw_adjust=1)
        sns.kdeplot(neg_num_nodes, label=f'Negative Num Nodes', fill=fill_b, color='blue', linestyle='--', alpha=1, bw_adjust=1)

        plt.title(f'Flex CNs versus. # of Nodes', fontsize=18)
        plt.ylabel('Density', fontsize=16)
        plt.xlabel('Score', fontsize=16)
        plt.xlim(left=-1)
        plt.legend(fontsize='x-large')
        plt.tight_layout()
        plt.savefig(test_path + f'Epoch{epoch+1}_CNvsNN_plot.png', dpi=300)
        plt.clf()

        sns.kdeplot(pos_scores[pos_scores > 35], label=f'Positive CNs', fill=fill_b, color='black', alpha=1, bw_adjust=1)
        sns.kdeplot(neg_scores[neg_scores > 35], label=f'Negative CNs', fill=fill_b, color='black', linestyle='--', alpha=1, bw_adjust=1)
        sns.kdeplot(pos_num_nodes[pos_num_nodes > 35], label=f'Positive Num Nodes', fill=fill_b, color='blue', alpha=1, bw_adjust=1)
        sns.kdeplot(neg_num_nodes[neg_num_nodes > 35], label=f'Negative Num Nodes', fill=fill_b, color='blue', linestyle='--', alpha=1, bw_adjust=1)

        plt.title(f'Flex CNs versus. # of Nodes (>35)', fontsize=18)
        plt.ylabel('Density', fontsize=16)
        plt.xlabel('Score', fontsize=16)
        plt.xlim(left=-1)
        plt.legend(fontsize='x-large')
        plt.tight_layout()
        plt.savefig(test_path + f'Epoch{epoch+1}_CNvsNNovr35_plot.png', dpi=300)
        plt.clf()

        # sns.kdeplot(plot_train, label=f'Pos. Train', fill=fill_b, color='orange', alpha=0.5, bw_adjust=1)
        # sns.kdeplot(plot_neg_train, label=f'Neg. Train', fill=fill_b, color='orange', alpha=0.5, bw_adjust=1)
        # sns.kdeplot(plot_val, label=f'Pos. Valid', fill=fill_b, color='red', alpha=0.5, bw_adjust=1)
        # sns.kdeplot(plot_neg_val, label=f'Neg. Valid', fill=fill_b, color='red', alpha=0.5, bw_adjust=1)
        # sns.kdeplot(plot_test, label=f'Pos. Test', fill=fill_b, color='blue', alpha=0.5, bw_adjust=1)
        # sns.kdeplot(plot_neg_test, label=f'Neg. Test', fill=fill_b, color='blue', alpha=0.5, bw_adjust=1)
        # sns.kdeplot(pos_scores, label=f'Pos. Train -- Epoch {epoch+1}', fill=fill_b, color='black', alpha=1, bw_adjust=1)
        # sns.kdeplot(neg_scores, label=f'Neg. Train -- Epoch {epoch+1}', fill=fill_b, color='black', linestyle='--', alpha=1, bw_adjust=1)

        # plt.title(f'Flex Generated {args.dataset} - Epoch {epoch+1} - CN', fontsize=18)
        # plt.ylabel('Density', fontsize=16)
        # plt.xlabel('Score', fontsize=16)
        # plt.xlim(left=-1)
        # plt.legend(fontsize='x-large')
        # plt.tight_layout()
        # plt.savefig(f'plot_cndist_Flex_Full_Epoch{epoch+1}_struct{args.struct}_{args.dataset}_{args.alpha}_{args.train_per}_max{args.max}_big{args.big}_J{args.J}.png', dpi=300)
        # plt.clf()


def get_roc_score(emb, edges_pos, edges_neg, gdc='bp'):
    def GraphDC(x):
        if gdc == 'ip':
            return 1 / (1 + np.exp(-x))
        elif gdc == 'bp':
            x = np.clip(x, None, 25)
            return 1 - np.exp( - np.exp(x))

    J = emb.shape[0]

    # Predict on test set of edges
    _, edges_pos = torch.unique(edges_pos, return_inverse=True)
    edges_pos = np.array(edges_pos).transpose((1,0))
    emb_pos_sp = emb[:, edges_pos[0], :]
    emb_pos_ep = emb[:, edges_pos[1], :]

    # preds_pos is torch.Tensor with shape [J, #pos_edges]
    preds_pos = GraphDC(
        np.einsum('ijk,ijk->ij', emb_pos_sp, emb_pos_ep)
    )
    
    if edges_neg.dim() > 2: edges_neg = edges_neg.view(-1, 2)
    _, edges_neg = torch.unique(edges_neg, return_inverse=True)
    edges_neg = np.array(edges_neg).transpose((1,0))
    emb_neg_sp = emb[:, edges_neg[0], :]
    emb_neg_ep = emb[:, edges_neg[1], :]

    preds_neg = GraphDC(
        np.einsum('ijk,ijk->ij', emb_neg_sp, emb_neg_ep)
    )

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(preds_pos.shape[-1]), np.zeros(preds_neg.shape[-1])])
    
    roc_score = np.array(
        [roc_auc_score(labels_all, pred_all.flatten()) \
            for pred_all in np.vsplit(preds_all, J)] 
    ).mean()
    
    ap_score = np.array(
        [average_precision_score(labels_all, pred_all.flatten()) \
            for pred_all in np.vsplit(preds_all, J)]
    ).mean()

    return roc_score, ap_score


def sigvae_loss(preds, labels, mu, logvar, emb, eps, n_nodes, norm, pos_weight):
    """
    Computing the negative ELBO for SIGVAE:
        loss = - \E_{h(z)} \log \frac{p(x|z)p(z)}{h(z)}.

    Parameters
    ----------
    preds : torch.Tensor of shape [J, N, N],
        Reconsurcted graph probability with J samples drawn from h(z).
    labels : torch.Tensor of shape [N, N],
        the ground truth connectivity between nodes in the adjacency matrix.
    mu : torch.Tensor of shape [K+J, N, zdim],
        the gaussian mean of q(z|psi).
    logvar : torch.Tensor of shape [K+J, N, zdim],
        the gaussian logvar of q(z|psi).
    emb: torch.Tensor of shape [J, N, zdim],
        the node embeddings that generate preds.
    eps: torch.Tensor of shape [J, N, zdim],
        the random noise drawn from N(0,1) to construct emb.
    n_nodes : int,
        the number of nodes in the dataset.
    norm : float,
        normalizing constant for re-balanced dataset.
    pos_weight : torch.Tensor of shape [1],
        stands for "positive weight", used for re-balancing +/- trainning samples.

    Returns
        reconstruction loss and kl regularizer.
    -------
    TYPE
        DESCRIPTION.

    """
    def get_rec(pred):
        # pred = torch.sigmoid(pred)
        # print("norm: ", norm)
        # print("pos_weight size: ", pos_weight.size())
        # print("labels.size: ", labels.size())
        # print("pred.size(): ", pred.size())
        log_lik = norm * (pos_weight * labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))  # N * N
        rec = -log_lik.mean()
        return rec

    
    # There are some problem with bce function when running bp models. Causes are under investigation.
    # def get_rec(pred):
    #     return norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight)


    # The objective is made up of 3 components,
    # loss = rec_cost + beta * (log_posterior - log_prior), where
    # rec_cost = -mean(log p(A|Z[1]), log p(A|Z[2]), ... log p(A|Z[J])),
    # log_prior = mean(log p(Z[1]), log p(Z[2]), ..., log p(Z[J])),
    # log_posterior = mean(log post[1], log post[2], ..., log post[J]), where
    # log post[j] = 1/(K+1) {q(Z[j]\psi[j]) + [q(Z[j]|psi^[1]) + ... + q(Z[j]|psi^[k])]}.
    # In practice, the loss is computed as
    # loss = rec_lost + log_posterior_ker - log_prior_ker.


    SMALL = 1e-6
    std = torch.exp(0.5 * logvar)
    J, N, zdim = emb.shape
    K = mu.shape[0] - J

    mu_mix, mu_emb = mu[:K, :], mu[K:, :]
    std_mix, std_emb = std[:K, :], std[K:, :]

    preds = torch.clamp(preds, min=SMALL, max=1-SMALL)

    # compute rec_cost
    rec_costs = torch.stack(
            [get_rec(pred) for pred in torch.unbind(preds, dim=0)],
            dim=0)
    # average over J * N * N items
    rec_cost = rec_costs.mean()



    # compute log_prior_ker, the constant 1/sqrt(2*pi) is cancelled out.
    # average over J items
    log_prior_ker = torch.sum(- 0.5 * emb.pow(2), dim=[1,2]).mean()


    # compute log_posterior
    # Z.shape = [J, 1, N, zdim]
    Z = emb.view(J, 1, N, zdim)

    # mu_mix.shape = std_mix.shape = [1, K, N, zdim]
    mu_mix = mu_mix.view(1, K, N, zdim)
    std_mix = std_mix.view(1, K, N, zdim)
    
    # compute -log std[k] - (Z[j] - mu[k])^2 / 2*std[k]^2 for all (j,k)
    # the shape of result tensor log_post_ker_JK is [J,K]
    log_post_ker_JK = - torch.sum(
        0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2,-1]
    )

    log_post_ker_JK += - torch.sum(
        (std_mix + SMALL).log(), dim=[-2,-1]
    )

    # compute -log std[j] - (Z[j] - mu[j])^2 / 2*std[j]^2 for j = 1,2,...,J
    # the shape of result tensor log_post_ker_J is [J, 1]
    log_post_ker_J = - torch.sum(
        0.5 * eps.pow(2), dim=[-2,-1]
    )
    log_post_ker_J += - torch.sum(
        (std_emb + SMALL).log(), dim = [-2,-1]
    )
    log_post_ker_J = log_post_ker_J.view(-1,1)


    # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1].
    log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)

    # apply "log-mean-exp" to the above tensor
    log_post_ker -= np.log(K + 1.) / J
    # average over J items.
    log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()


    #print(f"Rec: {rec_cost:.4f} | KL: {(log_posterior_ker - log_prior_ker):.4f} | Prior: {log_prior_ker:.4f} | Posterior: {log_posterior_ker:.4f}")

    return rec_cost, log_prior_ker, log_posterior_ker 

def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    edge_index = edge_index.cpu()
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
        # print('max cn: ', np.concatenate(scores, 0).max())

    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index

def SP(A, edge_index, remove=True):
    
    scores = []
    G = nx.from_scipy_sparse_array(A)
    print(len(G.edges()))
    add_flag1 = 0
    add_flag2 = 0
    count = 0
    count1 = count2 = 0
    print('remove: ', remove)
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        if nx.has_path(G, source=s, target=t):
            sp = nx.shortest_path_length(G, source=s, target=t)
        else:
            sp = 999
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
    
        scores.append(1/(sp))
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores), edge_index

def RA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / (A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

def PA(A, edge_index, batch_size=100000):
    # D. Liben-Nowell, J. Kleinberg. The Link Prediction Problem for Social Networks (2004). http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    
    G = nx.from_scipy_sparse_array(A)
    G_degree = np.array(G.degree(np.array(G.nodes())))
    edge_index = edge_index.cpu()
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        temp_tup = list(zip(list(src), list(dst)))
        cur_scores = G_degree[src][:, 1] * G_degree[dst][:, 1]
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

def get_flex_train_negs(split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None):
    """
    for some inexplicable reason ogb datasets split_edge object stores edge indices as (n_edges, 2) tensors
    @param split_edge:

    @param edge_index: A [2, num_edges] tensor
    @param num_nodes:
    @param num_negs: the number of negatives to sample for each positive
    @return: A [num_edges * num_negs, 2] tensor of negative edges
    """
   
      # any source is fine
    pos_edge = split_edge.t()
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(
        new_edge_index, num_nodes=num_nodes,
        num_neg_samples=pos_edge.size(1) * num_negs)
    return neg_edge.t()

def get_ogb_train_negs(split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None):
    """
    for some inexplicable reason ogb datasets split_edge object stores edge indices as (n_edges, 2) tensors
    @param split_edge:

    @param edge_index: A [2, num_edges] tensor
    @param num_nodes:
    @param num_negs: the number of negatives to sample for each positive
    @return: A [num_edges * num_negs, 2] tensor of negative edges
    """
   
      # any source is fine
    pos_edge = split_edge['train']['edge'].t()
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(
        new_edge_index, num_nodes=num_nodes,
        num_neg_samples=pos_edge.size(1) * num_negs)
    return neg_edge.t()

def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
        
def save_model(model, save_path, emb=None):

    if emb == None:
        state = {
            'state_dict_model'	: model.state_dict(),
            # 'state_dict_predictor'	: linkPredictor.state_dict(),
        }

    else:
        state = {
            'state_dict_model'	: model.state_dict(),
            'emb'	: emb.weight
        }

    print(str(save_path))
    torch.save(state, str(save_path) + "_mdl.pt")

def save_emb(score_emb, save_path):

    if len(score_emb) == 6:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x1, x2= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x1,
        'node_emb_with_valid_edges': x2

        }
        
    elif len(score_emb) == 5:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x
        }
    elif len(score_emb) == 4:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, = score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        }
   
    print(str(save_path))
    torch.save(state, str(save_path) + "_emb.pt")

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')

            r = best_result[:, 0].float()
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1].float()
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 2].float()
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 3].float()
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]


            return best_valid, best_valid_mean, mean_list, var_list

def get_logger(name, log_dir, config_dir):
	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def prep_data(data, edge_split):
    """
    Various prep
    """
    data.adj_t = data.adj_t.coalesce().bool().float()
    data.adj_t = data.adj_t.to_symmetric()

    train_edge_index = to_undirected(edge_split['train']['edge'].t())

    val_edge_index = to_undirected(edge_split['valid']['edge'].t())
    full_edge_index = torch.cat([train_edge_index, val_edge_index], dim=-1)

    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
    train_edge_weight = torch.ones([train_edge_index.size(1), 1], dtype=torch.float)
    full_edge_weight = torch.cat([train_edge_weight, val_edge_weight], 0).view(-1)

    data.full_edge_index = full_edge_index
    data.full_edge_weight = full_edge_weight
    data.full_adj = SparseTensor.from_edge_index(full_edge_index, full_edge_weight, [data.num_nodes, data.num_nodes])
    data.full_adj = data.full_adj.to_symmetric()

    return data

def calc_CN(data, use_val=False):
    """
    Calc CNs for all node pairs
    """
    if use_val:
        adj = data.full_adj_t
    else:
        adj = data.adj_t

    cn_scores = adj @ adj

    return cn_scores

def calc_PA(data, batch_size=100000):
    # D. Liben-Nowell, J. Kleinberg. The Link Prediction Problem for Social Networks (2004). http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    
    G_degree = degree(data.edge_index[0], data.num_nodes)

    return G_degree

def rank_score_matrix(row):
    """
    Rank from largest->smallest
    """
    num_greater_zero = (row > 0).sum().item()

    # Ignore 0s and -1s in ranking
    # Note: default is smallest-> largest so reverse
    if num_greater_zero > 0:
        ranks_row = rankdata(row[row > 0], method='min')
        ranks_row = ranks_row.max() - ranks_row + 1
        max_rank = ranks_row.max()
    else:
        ranks_row = []
        max_rank = 0

    # Overwrite row with ranks
    # Also overwrite 0s with max+1 and -1s with max+2
    row[row > 0] = ranks_row
    row[row == 0] = max_rank + 1
    row[row < 0] = max_rank + 2

    return row

def rank_and_merge_node(node_scores, true_pos_mask, data):
    """
    Do so for a single node
    """
    k = 250 // 2 

    # Nodes that are 0 for all scores. Needed later when selecting top K
    zero_nodes_score_mask = (node_scores == 0).numpy()

    # Individual ranks
    node_ranks = rank_score_matrix(node_scores.numpy())

    # If enough non-zero scores we use just take top-k
    # Otherwise we have to randomly select from 0 scores        
    max_greater_zero = data['num_nodes'] - zero_nodes_score_mask.sum().item() - true_pos_mask.sum().item()

    # NOTE: Negate when using torch.topk since 1=highest
    if max_greater_zero >= k:
        node_topk = torch.topk(torch.from_numpy(-node_ranks), k).indices
        node_topk = node_topk.numpy()
    elif max_greater_zero <= 0:
        # All scores are either true_pos or 0
        # We just sample from 0s here
        node_zero_score_ids = zero_nodes_score_mask.nonzero()[0]
        node_topk = np.random.choice(node_zero_score_ids, k)
    else:
        # First just take whatever non-zeros there are
        node_greater_zero = torch.topk(torch.from_numpy(-node_ranks), max_greater_zero).indices
        node_greater_zero = node_greater_zero.numpy()

        # Then choose the rest randomly from 0 scores
        node_zero_score_ids = zero_nodes_score_mask.nonzero()[0]
        node_zero_rand = np.random.choice(node_zero_score_ids, k-max_greater_zero)
        node_topk = np.concatenate((node_greater_zero, node_zero_rand))

    return node_topk.reshape(-1, 1)

def rank_and_merge_edges(edges, cn_scores, pa_scores, data, train_nodes, test=False):
    """
    For each edge we get the rank for the types of scores for each node and merge them together to one rank

    Using that we get the nodes with the top k ranks
    """
    all_topk_edges = []
    k = 250 // 2 

    # Used to determine positive samples to filter
    # For testing we also include val samples in addition to train
    if test:
        adj = data.full_adj
    else:
        adj = data.adj_t

    # if args.metric.upper() == "SP":
    #     edge_index, edge_weight = data.full_edge_index, data.full_edge_weight
    #     A_ssp = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(data.num_nodes, data.num_nodes))
    #     G = nx.from_scipy_sparse_array(A_ssp)    

    ### Get nodes not in train
    all_nodes = set(list(range(data.num_nodes)))
    nodes_not_in_train = torch.Tensor(list(all_nodes - train_nodes)).long()
    
    for edge in tqdm(edges, "Ranking Scores"):
        source, target = edge[0].item(), edge[1].item()

        source_adj = adj[source].to_dense().squeeze(0).bool()
        target_adj = adj[target].to_dense().squeeze(0).bool()

        #if args.metric.upper() == "CN":
        source_scores = cn_scores[source].to_dense().squeeze(0)
        target_scores = cn_scores[target].to_dense().squeeze(0)
        # elif args.metric.upper() == "PA":
        #     source_scores = target_scores = pa_scores
        # else:
        #     raise NotImplementedError(f"{arg.metric.upper()} is not implemented!")

        source_true_pos_mask = source_adj
        target_true_pos_mask = target_adj

        # Don't remove true positive
        # So just set all to 0
        # if args.keep_train_val:
        #     source_true_pos_mask = torch.zeros_like(source_true_pos_mask)
        #     target_true_pos_mask = torch.zeros_like(target_true_pos_mask)

        # Mask nodes not in train
        source_true_pos_mask[nodes_not_in_train] = 1
        target_true_pos_mask[nodes_not_in_train] = 1

        # Include masking for self-loops
        source_true_pos_mask[source], source_true_pos_mask[target] = 1, 1
        target_true_pos_mask[target], target_true_pos_mask[source] = 1, 1

        # Filter samples by setting to -1
        source_scores[source_true_pos_mask], source_scores[source_true_pos_mask] = -1, -1 

        source_topk_nodes = rank_and_merge_node(source_scores, source_true_pos_mask, data)
        source_topk_edges = np.concatenate((np.repeat(source, k).reshape(-1, 1), source_topk_nodes), axis=-1)

        target_topk_nodes = rank_and_merge_node(target_scores, target_true_pos_mask, data)
        target_topk_edges = np.concatenate((target_topk_nodes, np.repeat(target, k).reshape(-1, 1)), axis=-1)
        
        edge_samples = np.concatenate((source_topk_edges, target_topk_edges))
        all_topk_edges.append(edge_samples)

    return np.stack(all_topk_edges)


def calc_all_heuristics(data, split_edge, dataset_name):
    """
    Calc and store top-k negative samples for each sample
    """
    print("Prepping data...")
    data = prep_data(data, split_edge)

    # Get unique nodes in train
    train_nodes = set(split_edge['train']['edge'].flatten().tolist())

    print("Compute CNs...")
    cn_scores = calc_CN(data)
    print("Compute PA...")
    pa_scores = calc_PA(data)

    print("\n>>> Valid")
    val_neg_samples = rank_and_merge_edges(split_edge['valid']['edge'], cn_scores, pa_scores, data, train_nodes)
    with open(f"dataset/{dataset_name}Dataset/heart_valid_samples.npy", "wb") as f:
        np.save(f, val_neg_samples)

    print("\n>>> Test")
    test_neg_samples = rank_and_merge_edges(split_edge['test']['edge'], cn_scores, pa_scores, data, train_nodes, test=True)
    with open(f"dataset/{dataset_name}Dataset/heart_test_samples.npy", "wb") as f:
        np.save(f, test_neg_samples)

def find_k(train_dataset, dynamic_train, k=0.6):
    if train_dataset is None:
        k = 30
    else:
        if dynamic_train:
            sampled_train = train_dataset[:1000]
        else:
            sampled_train = train_dataset
        
        tracking = 0
        num_nodes = []
        for i in sampled_train:
            tracking = tracking + 1
            if tracking == 1000:
                break #DEBUG: Fix for off-by-one index error 
            
            num_nodes.append(i.num_nodes)
        num_nodes = sorted(num_nodes)

        k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
        k = max(10, k)
    
    return k

def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0, 
                   max_nodes_per_hop=None, node_features=None, 
                   y=1, directed=False, A_csc=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A. 
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes].tolil()

    # Remove target link between the subgraph.
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0
    subgraph = subgraph.tocsr()

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)

def compute_cn_mask(adj):
    """
    Returns CN and 1-Hop (non-CN / XOR) nodes for all
    links in a batch

    adj: torch.sparse adjacency matrix (|V| x |V|)
    """

    adj = adj.tocoo()
    batch = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)
    # Yes this is a lot of type conversion but it runs fast and works
    adj = torch.sparse.LongTensor(torch.LongTensor(np.array([adj.row, adj.col])),
                              torch.LongTensor(adj.data.astype(np.int32)))
    
    assert adj.shape[0] == adj.shape[1], "adj not symmetric after sparsetensor in CN embed compute"
    
    # np_adj = adj.to_dense().numpy() # DEBUG -- Visualizations
    # G = nx.from_numpy_array(np_adj)
    # plt.figure(figsize=(8, 6))
    # pos = nx.spring_layout(G) 
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    # plt.title("Graph Visualization")
    # plt.savefig("TEST_FLEX.png") # END DEBUG

    # Rows corresponding to both nodes in edges
    src_adj = torch.index_select(adj, 0, batch[0])
    tgt_adj = torch.index_select(adj, 0, batch[1])

    # Equals: {0: ">1-Hop", 1: "1-Hop (Non-CN)", 2: "CN"}
    pair_adj = src_adj + tgt_adj 

    # Converting to a COO representation removes >1-Hop entries
    pair_ix  = pair_adj.coalesce().indices() # TODO: Bad bottleneck on larger datasets/percent splits/hops
    pair_vals = pair_adj.coalesce().values()

    # Get indices for appropriate nodes
    # Both are of shape [2, *] where...
    # [0, :] = Batch ID and [1, :] = Node ID
    #xor_nodes = pair_ix[:, pair_vals == 1] # DEBUG -- XOR only necessary if bad things happen with just CN
    cn_nodes = pair_ix[:, pair_vals == 2]

    cn_nodes = cn_nodes.flatten()
    #xor_nodes = xor_nodes.flatten()
    batch = batch.flatten()
    
    unq_cn = torch.unique(cn_nodes[torch.isin(cn_nodes, batch)])
    #xor_cn = torch.unique(xor_nodes[torch.isin(xor_nodes, batch)])
    labels = torch.zeros(int(torch.max(batch)) + 1, dtype=torch.int64)
    
    z = labels.index_fill(dim=0, index=unq_cn, value=1)

    adj = adj.coalesce().bool().float()
    assert adj.shape[0] == adj.shape[1], "Adj not symmetric after coalesce in CN compute"
    
    return z, adj


def construct_pyg_graph(node_ids, adj, dists, node_features, y, src, dst, node_label='drnl'):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]
    
    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    

    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
        adj = adj.tocoo()
        adj = torch.sparse.LongTensor(torch.LongTensor(np.array([adj.row, adj.col])),
                                torch.LongTensor(adj.data.astype(np.int32)))
        adj_t = adj.coalesce().bool().float()
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists)==0).to(torch.long)
        adj = adj.tocoo()
        adj = torch.sparse.LongTensor(torch.LongTensor(np.array([adj.row, adj.col])),
                                torch.LongTensor(adj.data.astype(np.int32)))
        adj_t = adj.coalesce().bool().float()
    elif node_label == 'cn':
        z, adj_t = compute_cn_mask(adj)
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z, 
                node_id=node_ids, num_nodes=num_nodes, adj_t=adj_t, edge=torch.tensor([[src, dst]]))
    return data

 
def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl', 
                                ratio_per_hop=1.0, max_nodes_per_hop=None, 
                                directed=False, A_csc=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop, 
                             max_nodes_per_hop, node_features=x, y=y, 
                             directed=directed, A_csc=A_csc)
        ##tmp = nodes, subgraph, dists, node_features, y
        data = construct_pyg_graph(*tmp, src, dst, node_label)
        data_list.append(data)

    return data_list


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge

def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, struct, percent=100):
    pos_edge = split_edge[split]['edge'].t()
    
    # and not struct
    if split == 'train':
        if struct:
            train_edge = split_edge['train']['edge']
            train_years = split_edge['train']['year']
            year_mask = (train_years >= 2016) & (train_years <= 2017)
            pos_edge = train_edge[year_mask].t()
            edge_index = pos_edge
            print(f"Structured Sampling Size: {pos_edge.size()}", flush=True)
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
            new_edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1))
    elif split_edge[split]['edge_neg'].dim() == 2:
        neg_edge = split_edge[split]['edge_neg'].t()
    else:
        neg_edge = split_edge[split]['edge_neg']
        neg_edge = torch.permute(neg_edge, (2, 0, 1))
        neg_edge = neg_edge.view(2,-1)
        print(f"Sampled {split} Edge Negative: {neg_edge.size()}")
    # elif struct and split == 'train':
    #     print("Structured Negative Sampling", flush=True)
    #     new_edge_index, _ = add_self_loops(edge_index)
    #     neg_edge = negative_sampling(
    #         new_edge_index, num_nodes=num_nodes,
    #         num_neg_samples=pos_edge.size(1)*1000000) # arbitrary # of negatives to align positives when sorting
    
    
    # subsample for pos_edge
    np.random.seed(123)
    num_pos = pos_edge.size(1)
    perm = np.random.permutation(num_pos)
    perm = perm[:int(percent / 100 * num_pos)]
    pos_edge = pos_edge[:, perm]

    # subsample for neg_edge
    np.random.seed(123)
    num_neg = neg_edge.size(1)
    perm = np.random.permutation(num_neg)
    perm = perm[:int(percent / 100 * num_neg)]
    neg_edge = neg_edge[:, perm]

    if neg_edge.size(1) % pos_edge.size(1) != 0:
        neg_edge = neg_edge[:, -250*pos_edge.size(1) :] #slice to 250 negatives per positive


    return pos_edge, neg_edge