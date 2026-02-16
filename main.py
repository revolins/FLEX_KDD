
import sys
sys.path.append("..") 

import torch
import numpy as np
import argparse
import scipy.sparse as sp
from ncn_model import predictor_dict, convdict, NCN_GCN
from utils import *
from eval import *

from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx, to_undirected, to_edge_index
from functools import partial

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from flex_dataset import SynthDataset, FlexDataset, FlexDynamicDataset

from torch_geometric.utils import negative_sampling
import os
import torch_geometric.transforms as T

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
log_print = get_logger('testrun', 'log', ROOT_DIR)

def build_vgae(args, input_dim):
    return GCNModelSIGVAE(
        32, input_dim, hidden_dim1=32, hidden_dim2=16, dropout=0.0,
        encsto='semi', 
        gdc='bp', 
        ndist='Bernoulli',
        copyK=args.K, 
        copyJ=args.J, 
        device=args.device
    )

def flex_train(model, gnn, score_func, optimizer, train_loader, epoch, alpha, device, train_data_len, args, cnprobs, ncn_alpha):

    model.train()
    gnn.train()
    score_func.train()
    final_loss, cfs_loss, rls_loss = 0, 0, 0
    pos_scores, neg_scores, pos_num_nodes, neg_num_nodes = [], [], [], []

    for data in tqdm(train_loader, ncols=70, desc=f"Training Epoch: {epoch}"):
        data = data.to(device)
        optimizer.zero_grad()
        features = data.x
        struct_feat = data.z
        
        adj_coo = data.adj_t.coalesce()
        row, col, values = adj_coo.coo()
        if values == None: values = torch.ones(row.size(0), dtype=torch.float32)

        scipy_csr = sp.csr_matrix((values.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())), shape=(adj_coo.size(0), adj_coo.size(1)))

        adj_norm = preprocess_graph(scipy_csr).to(device)
        adj_label = scipy_csr + sp.eye(scipy_csr.shape[0])
        adj_label = torch.FloatTensor(adj_label.toarray())

        pos_weight = torch.tensor([float(scipy_csr.shape[0] * scipy_csr.shape[0] - scipy_csr.sum()) / (scipy_csr.sum() + 1.0)])
        norm = scipy_csr.shape[0] * scipy_csr.shape[0] / (float((scipy_csr.shape[0] * scipy_csr.shape[0] - scipy_csr.sum()) * 2) + 1.0)
        recovered, mu, logvar, z, z_scaled, eps, rk, snr = model(features, adj_norm, struct_feat, list(Counter(data.batch.cpu().numpy()).values()))

        pos_weight = pos_weight.to(device)
        adj_label = adj_label.to(device)

        loss_rec, loss_prior, loss_post = sigvae_loss(
            preds=recovered, 
            labels=adj_label,
            mu=mu, 
            logvar=logvar, 
            emb=z, 
            eps=eps, 
            n_nodes=data.num_nodes,
            norm=norm, 
            pos_weight=pos_weight
        )

        WU = 1.0
        target_kl = args.target_kl

        actual_kl = loss_post - loss_prior
        kl_reg = -((actual_kl - target_kl) ** 2)  # maximized when KL ≈ target
        loss_train = -(loss_rec + WU * kl_reg / (data.num_nodes**2))

        recon_adj = F.relu(torch.mean(recovered, dim=0))
        recon_adj[recon_adj < args.threshold] = 0.0
        recon_adj.fill_diagonal_(0.0)
        
        row, col = recon_adj.nonzero(as_tuple=True)

        sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(recon_adj.size(0), recon_adj.size(1)))
        sparse_adj = sparse_adj.to_symmetric()

        h = gnn(data.x, sparse_adj)        

        out = score_func.multidomainforward(h,
                                            sparse_adj,
                                            data.edge.t(),
                                            cndropprobs=cnprobs)

        pos_out = out.view(-1)[data.y.view(-1)==1]
        pos_loss = -F.logsigmoid(pos_out).mean()
        neg_out = out.view(-1)[data.y.view(-1)==0]
        neg_loss = -F.logsigmoid(neg_out).mean()
        cf_loss = pos_loss + neg_loss

        total_loss = loss_train + (alpha * cf_loss)
        
        total_loss.backward()

        optimizer.step()

        final_loss += total_loss.item() * data.num_graphs
        cfs_loss += (alpha * cf_loss.item()) * data.num_graphs
        rls_loss += loss_train.item() * data.num_graphs
    
    
    return (final_loss / train_data_len, cfs_loss / train_data_len, rls_loss / train_data_len), z_scaled, pos_scores, neg_scores, pos_num_nodes, neg_num_nodes


def main():
    parser = argparse.ArgumentParser(description='NCN')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--score_model', type=str, default='NCN')

    ### train setting
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=100,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_envs', type=int, default=3)
    parser.add_argument('--loss_coeff', type=float, default=0.1)
    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--remove_edge_aggre', action='store_true', default=False)
    
    ####### ncn
    parser.add_argument('--mplayers', type=int, default=3)
    parser.add_argument('--nnlayers', type=int, default=3)
    parser.add_argument('--ln', action="store_true")
    parser.add_argument('--lnnn', action="store_true")
    parser.add_argument('--res', action="store_true")
    parser.add_argument('--jk', action="store_true")
    parser.add_argument('--maskinput', action="store_true")
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--gnndp', type=float, default=0.3)
    parser.add_argument('--xdp', type=float, default=0.25)
    parser.add_argument('--tdp', type=float, default=0.05)
    parser.add_argument('--gnnedp', type=float, default=0.25)
    parser.add_argument('--predp', type=float, default=0.3)
    parser.add_argument('--preedp', type=float, default=0.0)
    parser.add_argument('--splitsize', type=int, default=-1)
    parser.add_argument('--gnnlr', type=float, default=0.0003)
    parser.add_argument('--prelr', type=float, default=0.0003)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--testbs', type=int, default=8192)
    parser.add_argument('--probscale', type=float, default=2.5)
    parser.add_argument('--proboffset', type=float, default=6.0)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--trndeg', type=int, default=-1)
    parser.add_argument('--tstdeg', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default="cora_CN_2_1_0")
    parser.add_argument('--predictor', choices=predictor_dict.keys())
    parser.add_argument('--model', choices=convdict.keys())
    parser.add_argument('--cndeg', type=int, default=-1)
    parser.add_argument('--save_gemb', action="store_true")
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--cnprob', type=float, default=0)
    parser.add_argument('--pt', type=float, default=0.1)
    parser.add_argument("--learnpt", action="store_true")
    parser.add_argument("--use_xlin", action="store_true")
    parser.add_argument("--tailact", action="store_true")
    parser.add_argument("--twolayerlin", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--increasealpha", action="store_true")
    parser.add_argument("--use_rand", action="store_true")

    # SIGVAE
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
    parser.add_argument('--dynamic', action='store_true',
                    help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--train_per', type=int, default=1)
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--node_label', type=str, default='zo')
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.9999)
    parser.add_argument('--target_kl', dest='target_kl', type=float, default=100.0,
            help='Learning rate.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=64,
            help='Number of workers to load data.')
    parser.add_argument('--K', type=int, default=15,
                    help='number of samples to draw for MC estimation of h(psi).')
    parser.add_argument('--J', type=int, default=20,
                    help='Number of samples to draw for MC estimation of log-likelihood.')
    parser.add_argument('--struct', action='store_true')

    args = parser.parse_args()
   
    print('use_val_edge:', args.use_valedges_as_input)
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if 'cn' in args.dataset.lower() or 'pa' in args.dataset.lower() or 'sp' in args.dataset.lower():
            dataset_name = args.dataset + '_seed1'
            print("################################")
            print(f'Loading Dataset: {dataset_name}')
            print("################################")
            data = SynthDataset(dataset_name=dataset_name).get()
            if 'ppa' in dataset_name: data.x = data.x.float()
             
            split_edge = SynthDataset(dataset_name=dataset_name).get_edge_split()
    else:

        dataset = PygLinkPropPredDataset(name=args.dataset, root=os.path.join(ROOT_DIR, "dataset", args.dataset))
    
        data = dataset[0]
        split_edge = dataset.get_edge_split()

    edge_index = data.edge_index
    
    data.max_x = -1
    if 'ddi' in args.dataset: 
        data.x = torch.arange(data.num_nodes)
        data.max_x = data.num_nodes
        data.x = data.x.to(torch.long)
    else: data.x = data.x.to(torch.float)
    
    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            edge_weight = data.edge_weight.to(torch.float)
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            train_edge_weight = split_edge['train']['weight']
            train_edge_weight = train_edge_weight.to(torch.float)
        else:
            train_edge_weight = None

    else:
        train_edge_weight = None

    data = T.ToSparseTensor()(data)
    if hasattr(data, 'adj_t'):
        row, col, value = data.adj_t.coo()
        if value is not None and value.dim() == 2 and value.size(1) == 1:
            value = value.squeeze(1)
            data.adj_t = SparseTensor(row=row, col=col, value=value,
                    sparse_sizes=(data.adj_t.size(0), data.adj_t.size(1)))
    data.edge_index = to_edge_index(data.adj_t)[0]
    data.edge_index = edge_index
    data.edge_index = data.edge_index

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)

        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float)
        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        print(edge_weight.size())
        A = SparseTensor.from_edge_index(full_edge_index, edge_weight.view(-1), [data.num_nodes, data.num_nodes])
        
        data.full_adj_t = A
        data.full_edge_index = full_edge_index
        print(data.full_adj_t)
        print(data.adj_t)

    predfn = predictor_dict[args.score_model]
    predfn = partial(predfn, use_xlin=args.use_xlin, tailact=args.tailact, twolayerlin=args.twolayerlin, beta=args.beta)

    model = NCN_GCN(in_channels=data.num_features, hidden_channels=args.hidden_channels, 
                out_channels=args.hidden_channels, num_layers=args.mplayers,
                    dropout=args.gnndp, ln=args.ln, res=args.res, max_x=data.max_x,
                    conv_fn=args.model, jk=args.jk, edrop=args.gnnedp, 
                    xdropout=args.xdp, taildropout=args.tdp, noinputlin=False).to(device)

    score_func = predfn(args.hidden_channels, args.hidden_channels, 1, args.nnlayers,
                args.predp, args.preedp, args.lnnn).to(device)
    
    vgae = build_vgae(args, data.num_features + 32).to(device)
    
    while split_edge['train']['edge'].size(0) <= args.batch_size:
        args.batch_size = args.batch_size // 2
        if args.batch_size <= 0:
            raise Exception("Batch Size Reached 0 in Pos. Train Edges")
        
    while split_edge['valid']['edge'].size(0) <= args.testbs:
        args.testbs = args.testbs // 2
        if args.testbs <= 0:
            raise Exception("Batch Size Reached 0 in Pos. Val. Edges")
        
    while split_edge['test']['edge'].size(0) <= args.testbs:
        args.testbs = args.testbs // 2
        if args.testbs <= 0:
            raise Exception("Batch Size reached 0 in Pos. Testing Edges")
        
    if args.use_rand:
            print("Loading Random Negatives")
            neg_valid_edge = split_edge['valid']['edge_neg']
            neg_test_edge = split_edge['test']['edge_neg']
    else:    
        print("Loading Heart Negatives")
        with open(f'dataset/{dataset_name}Dataset/heart_valid_samples.npy', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
            split_edge['valid']['edge_neg'] = neg_valid_edge
        with open(f'dataset/{dataset_name}Dataset/heart_test_samples.npy', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
            split_edge['test']['edge_neg'] = neg_test_edge
        
        print("print(split_edge['valid']['edge_neg'].size()) ", split_edge['valid']['edge_neg'].size())
        print("print(split_edge['test']['edge_neg'].size()) ", split_edge['test']['edge_neg'].size())

    if args.data_appendix == '':
        args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))

    path = 'dataset/'+ str(args.dataset)+ '_flex{}'.format(args.data_appendix) # Start FlexDataset Loading
        
    dataset_class = 'FlexDynamicDataset' if args.dynamic else 'FlexDataset'

    train_dataset = eval(dataset_class)(
        path, 
        data, 
        split_edge, 
        num_hops=args.num_hops, 
        percent=args.train_per, 
        split='train', 
        use_coalesce=False, 
        node_label=args.node_label, 
        ratio_per_hop=args.ratio_per_hop, 
        max_nodes_per_hop=None,
        run=0,
        add_val=args.use_valedges_as_input
    )

    train_dataset.data = train_dataset.data.to(device)
    train_data_len = len(train_dataset)

    split_edge['train']['edge_neg'] = split_edge['train']['edge_neg'].to(device)
    split_edge['train']['edge'] =  split_edge['train']['edge'].to(device)
    split_edge['valid']['edge_neg'] = split_edge['valid']['edge_neg'].to(device)
    split_edge['valid']['edge'] = split_edge['valid']['edge'].to(device)
    split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'].to(device)
    split_edge['test']['edge'] = split_edge['test']['edge'].to(device)
    data = data.to(device)

    if not args.dynamic:
        for j in range(train_data_len):
            train_dataset.data.adj_t[j] = train_dataset.data.adj_t[j].coalesce().bool().float() # Clamp edge_weights

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@5': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
        'AUC':Logger(args.runs),
        'AP':Logger(args.runs),
         'mrr_hit20':  Logger(args.runs),
        'mrr_hit50':  Logger(args.runs),
        'mrr_hit100':  Logger(args.runs),
    }

    if args.dataset =='ogbl-collab':
        eval_metric = 'Hits@50'
    else:
        eval_metric = 'Hits@20'

    best_valid_auc = best_test_auc = 2

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run + 1
            args.seed = seed
        
        print('seed: ', seed)
        init_seed(seed)
        model.reset_parameters()
        score_func.reset_parameters()

        if args.load:
            model.load_state_dict(torch.load(f'models/{args.dataset}_run{seed}_gnn.pt', map_location=device), strict=False)
            score_func.load_state_dict(torch.load(f'models/{args.dataset}_run{seed}_score_model.pt', map_location=device), strict=False)

        log_print.info(f"INITIAL NCN RESULTS on {args.dataset}")
        if args.dataset == 'ogbl-collab' or args.dataset == 'ogbl-ddi':
            print("Random Sample Testing")
            results_rank, _ = ncn_test(model, score_func, data, split_edge, evaluator_hit, args.testbs, args.use_valedges_as_input)
        else:
            print("Hard Sample Testing")
            results_rank, _ = ncn_hard_test(model, score_func, data, split_edge, evaluator_mrr, evaluator_hit, args.testbs, args.use_valedges_as_input)
        for key, result in results_rank.items():
                print(key)
                
                train_hits, valid_hits, test_hits = result
                log_print.info(
                    f'Eval Only Results from Seed: {seed:02d}, '
                        f'Train: {100 * train_hits:.2f}%, '
                        f'Valid: {100 * valid_hits:.2f}%, '
                        f'Test: {100 * test_hits:.2f}%')
        
        optimizer = torch.optim.Adam([{'params': model.parameters(), "lr": args.gnnlr}, 
           {'params': score_func.parameters(), 'lr': args.prelr}, {'params': vgae.parameters(), 'lr': args.gnnlr}])

        best_valid = 0
        kill_cnt = 0
        best_test = 0
        
        vgae.load_state_dict(torch.load(f'models/{args.dataset}_run{seed}_sigvae.pt', map_location=device))
        vgae = vgae.to(device)
        log_print.info(f'SIGVAE returned: {vgae}')

        for epoch in range(1, 1 + args.epochs):
            loss, _, _, _, _, _ = flex_train(vgae, model, score_func, optimizer, train_loader, epoch, args.alpha, device, train_data_len, args, [], None)
           
            if epoch % args.eval_steps == 0:

                total_loss, cf_loss, rl_loss = loss

                log_print.info(
                    '***** VGAE *****, '
                    f'Run: {run:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Total Loss: {total_loss:.4f}, '
                    f'CF Loss: {cf_loss:.4f}, '
                    f'RL Loss: {rl_loss:.4f}, ')

                if args.dataset == 'ogbl-collab' or args.dataset == 'ogbl-ddi':
                    print("Random Sample Testing")
                    results_rank, _ = ncn_test(model, score_func, data, split_edge, evaluator_hit, args.testbs, args.use_valedges_as_input)
                else:
                    results_rank, _ = ncn_hard_test(model, score_func, data, split_edge, evaluator_mrr, evaluator_hit, args.testbs, args.use_valedges_as_input)

                
                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)


                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {total_loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')

                r = torch.tensor(loggers[eval_metric].results[run])
                best_valid_current = round(r[:, 1].max().item(),4)
                best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                print(eval_metric)
                log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                f'best test: {100*best_test:.2f}%')
                
                if len(loggers['AUC'].results[run]) > 0:
                    r = torch.tensor(loggers['AUC'].results[run])
                    best_valid_auc = round(r[:, 1].max().item(), 4)
                    best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)
                    
                    print('AUC')
                    log_print.info(f'best valid: {100*best_valid_auc:.2f}%, '
                                f'best test: {100*best_test_auc:.2f}%')
                
                print('---')
                
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                else:
                    kill_cnt += 1
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            if len(loggers[key].results[0]) > 0:
                print(key)
                loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():
        if len(loggers[key].results[0]) > 0:
            print(key)
            
            best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()
            result_all_run[key] = [mean_list, var_list]
    
if __name__ == "__main__":

    main()


    
