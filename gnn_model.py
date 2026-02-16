import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as tdist
import numpy as np

from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import negative_sampling, to_torch_csr_tensor
from torch_geometric.nn.inits import reset
from torch.autograd import Function
from torch.nn import (LayerNorm, Embedding, Module)
from torch_sparse import SparseTensor, masked_select_nnz
from torch_sparse.matmul import spmm_add
from typing import Final, Optional, Tuple, Any, Iterable

EPS = 1e-15
MAX_LOGSTD = 5

# SIG-VAE code adapted from github: https://github.com/YH-UtMSB/sigvae-torch
# Credit to Yilin He for adapting Hasanzadeh et al's SIG-VAE code from Tensorflow to PyTorch
def build_vgae(args, input_dim, model):
    return GCNModelSIGVAE(
        32, input_dim, hidden_dim1=32, hidden_dim2=16, dropout=0.0,
        encsto='semi', 
        gdc='bp', 
        ndist='Bernoulli',
        copyK=args.K, 
        copyJ=args.J, 
        device=args.device
    )

class DropAdj(torch.nn.Module):
    doscale: Final[bool]
    def __init__(self, dp: float = 0.0, doscale=True, drop_node=False) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale
        self.drop_node = drop_node

    def forward(self, adj: SparseTensor, x=None)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = masked_select_nnz(adj, mask, layout="coo") 
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        if self.drop_node: return adj, x[mask]
        return adj

class xGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, edge_drop, max_z, x):
        super(xGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        self.adjdrop = DropAdj(edge_drop)

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
            self.lins.append(LayerNorm(hidden_channels))

        elif num_layers > 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.lins.append(LayerNorm(hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels))
                self.lins.append(LayerNorm(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     
    def forward(self, x, edge_index, z=None):

        #edge_index = to_torch_csr_tensor(edge_index, size=(x.size(0), x.size(0)))
            
        for conv, lin in zip(self.convs[:-1], self.lins):
            x = conv(x, edge_index)
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        return x

class mlp_edge(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(mlp_edge, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i=None, x_j=None):
        x = x_i * x_j
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class mlp_score(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(mlp_score, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i=None, x_j=None, temp_x=None, batch=None):
        if batch is not None:
            x = global_mean_pool(temp_x, batch)
        else:
            x = x_i * x_j
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class mlp_adv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(mlp_adv, self).__init__()

        self.lins = torch.nn.ModuleList()

        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))

            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.invest = 1
        self.num_layers = num_layers

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t=None):
        if self.invest == 1:
            self.invest = 0
       
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)

        return x


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class unGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, edge_drop, ood_method):
        super(unGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.adjdrop = DropAdj(edge_drop)
        self.ood_method = ood_method

        if ood_method == 'DANN':
            self.grl = GradientReverseLayer()
            #in_channels, hidden_channels, out_channels, num_layers,dropout
            self.d_c = mlp_adv(hidden_channels, hidden_channels, 1, 3, 
                                            dropout)
        
        if ood_method == 'IRM':
            self.dummy_w = torch.nn.Parameter(torch.Tensor([1.0]))

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))

        elif num_layers > 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t, z=None):
        adj_t_drop = self.adjdrop(adj_t)
        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t_drop)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t_drop)

        if self.ood_method == 'DANN':
            adv_x = self.grl(x)
            adv_pred = self.d_c(adv_x)

            return x, adv_pred
        
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, edge_drop, ood_method):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.adjdrop = DropAdj(edge_drop)
        self.ood_method = ood_method

        if ood_method == 'DANN':
            self.grl = GradientReverseLayer()
            #in_channels, hidden_channels, out_channels, num_layers,dropout
            self.d_c = mlp_adv(hidden_channels, hidden_channels, 1, 3, 
                                            dropout)
        
        if ood_method == 'IRM':
            self.dummy_w = torch.nn.Parameter(torch.Tensor([1.0]))

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
            self.lins.append(LayerNorm(hidden_channels))

        elif num_layers > 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.lins.append(LayerNorm(hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels))
                self.lins.append(LayerNorm(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
        # self.p = args
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     

    def forward(self, x, adj_t, z=None):
        adj_t_drop = self.adjdrop(adj_t)
        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv, lin in zip(self.convs[:-1], self.lins):
            x = conv(x, adj_t_drop)
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t_drop)

        if self.ood_method == 'DANN':
            adv_x = self.grl(x)
            adv_pred = self.d_c(adv_x)

            return x, adv_pred
        
        return x

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()
        self.max_z = 2
        self.z_embedding = Embedding(self.max_z, out_channels)
        self.convs = torch.nn.ModuleList()

        print("in channels: ", in_channels)
        print("out_channels: ", out_channels)
        print("num_layers: ", num_layers)

        self.conv1 = GCNConv(in_channels, 2 * out_channels)

        if num_layers >= 3:
            for _ in range(num_layers - 3):
                self.convs.append(GCNConv(2 * out_channels, 2 * out_channels))
                
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, z=None, ablate=False):
        if not ablate:
            z_emb = self.z_embedding(z)
            x = torch.cat([x, z_emb.to(torch.float)], dim=1)

        x = self.conv1(x, edge_index).relu()
        for conv in self.convs[:-2]:
            x = conv(x, edge_index).relu()

        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# Source Code adapted from: https://github.com/YH-UtMSB/sigvae-torch
# Credit to Yilin He for adapting Hasanzadeh et al's code from Tensorflow to PyTorch

class GraphConvolution(Module):
    """
    GCN layer, based on https://arxiv.org/abs/1609.02907
    that allows MIMO
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)

        """
        if the input features are a matrix -- excute regular GCN,
        if the input features are of shape [K, N, Z] -- excute MIMO GCN with shared weights.
        """
        # An alternative to derive XW (line 32 to 35)
        # W = self.weight.view(
        #         [1, self.in_features, self.out_features]
        #         ).expand([input.shape[0], -1, -1])
        # support = torch.bmm(input, W)

        tem_l = [torch.mm(inp, self.weight) for inp in torch.unbind(input, dim=0)]

        support = torch.stack(
                tem_l,
                dim=0)

        output = torch.stack(
                [torch.spmm(adj, sup) for sup in torch.unbind(support, dim=0)],
                dim=0)

        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
class GCNModelSIGVAE(nn.Module):
    def __init__(self, ndim, input_feat_dim, hidden_dim1, hidden_dim2, dropout=0.0, encsto='semi', gdc='ip', ndist = 'Bernoulli', copyK=1, copyJ=1, device='cuda', dataset=None, max_x=None):
        super(GCNModelSIGVAE, self).__init__()

        self.max_z = 1000
        self.z_embedding = Embedding(self.max_z, hidden_dim1)

        self.gce = GraphConvolution(ndim, hidden_dim1, dropout, act=F.relu)
        # self.gc0 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.encsto = encsto
        self.dc = GraphDecoder(hidden_dim2, dropout, gdc=gdc)
        self.device = device
        if dataset=='ogbl-ddi':
            tmp = nn.Embedding(max_x + 1, hidden_dim1)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_dim1

        if ndist == 'Bernoulli':
            self.ndist = tdist.Bernoulli(torch.tensor([.5], device=self.device))
        elif ndist == 'Normal':
            self.ndist = tdist.Normal(
                    torch.tensor([0.], device=self.device),
                    torch.tensor([1.], device=self.device))
        elif ndist == 'Exponential':
            self.ndist = tdist.Exponential(torch.tensor([1.], device=self.device))

        # K and J are defined in http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # Algorthm 1.
        self.K = copyK
        self.J = copyJ
        self.ndim = ndim
        if dataset == 'ogbl-ddi':
            self.xemb = nn.Sequential(nn.Dropout(0.0))
            self.xemb.append(nn.Linear(input_feat_dim, hidden_dim1))

        # parameters in network gc1 and gce are NOT identically distributed, so we need to reweight the output 
        # of gce() so that the effect of hiddenx + hiddene is equivalent to gc(x || e).
        self.reweight = ((self.ndim + hidden_dim1) / (input_feat_dim + hidden_dim1))**(.5)

    def reset_parameters(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def encode(self, x, adj, z=None, ablate=False):
        if x.dim() == 1: x = self.xemb(x)
        if z != None and not ablate:
            z_emb = self.z_embedding(z)
            x = torch.cat([x, z_emb.to(torch.float)], dim=1)
        #print("x size: ", x.size())
        x = torch.unsqueeze(x, 0)

        assert len(x.shape) == 3, 'The input tensor dimension is not 3!'
        # Without torch.Size(), an error would occur while resampling.

        hiddenx = self.gc1(x, adj)

        if self.ndim >= 1:
            e = self.ndist.sample(torch.Size([self.K+self.J, x.shape[1], self.ndim]))
            e = torch.squeeze(e, -1)
            e = e.mul(self.reweight)

            hiddene = self.gce(e, adj)
        else:
            print("no randomness.")
            hiddene = torch.zeros(self.K+self.J, hiddenx.shape[1], hiddenx.shape[2], device=self.device)
        
        hidden1 = hiddenx + hiddene

        # hiddens = self.gc0(x, adj)

        p_signal = hiddenx.pow(2.).mean()
        p_noise = hiddene.pow(2.).mean([-2,-1])
        snr = (p_signal / p_noise)
        

        # below are 3 options for producing logvar
        # 1. stochastic logvar (more instinctive)
        #    where logvar = self.gc3(hidden1, adj)
        #    set args.encsto to 'full'.
        # 2. deterministic logvar, shared by all K+J samples, and share a previous hidden layer with mu
        #    where logvar = self.gc3(hiddenx, adj)
        #    set args.encsto to 'semi'.
        # 3. deterministic logvar, shared by all K+J samples, and produced by another branch of network
        #    (the one applied by A. Hasanzadeh et al.)
        

        mu = self.gc2(hidden1, adj)

        EncSto = (self.encsto == 'full')
        hidden_sd = EncSto * hidden1 + (1 - EncSto) * hiddenx
        
        logvar = self.gc3(hidden_sd, adj)
        
        return mu, logvar, snr

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std)
        if not self.training:
            return mu, eps
        return eps.mul(std).add(mu), eps

    def forward(self, x, adj, data_z, train_nodes, ablate=False):
        if data_z == None: AssertionError("must pass data.z or structural features")
        
        mu, logvar, snr = self.encode(x, adj, data_z, ablate)
        
        emb_mu = mu[self.J:, :]
        emb_logvar = logvar[self.J:, :]

        # check tensor size compatibility
        assert len(emb_mu.shape) == len(emb_logvar.shape), 'mu and logvar are not equi-dimension.'
        
        split_emb_mu = torch.split(emb_mu, train_nodes, dim=1)
        split_emb_logvar = torch.split(emb_logvar, train_nodes, dim=1)

        num_nodes = sum(train_nodes)
        adj_global = torch.zeros(self.K, num_nodes, num_nodes, device=mu.device)
        z_global = torch.zeros(self.K, num_nodes, mu.size(2), device=mu.device)
        zsc_global = torch.zeros(self.K, num_nodes, mu.size(2), device=mu.device)
        eps_global = torch.zeros(self.K, num_nodes, mu.size(2), device=mu.device)
        start_idx = 0

        for i, _ in enumerate(train_nodes):
            z, eps = self.reparameterize(split_emb_mu[i], split_emb_logvar[i])

            adj_, z_scaled, rk = self.dc(z)
            node_indices = torch.arange(start_idx, start_idx + train_nodes[i])

            #rk stays the same for each subgraph size - (1, HidDim)
            adj_global[:, node_indices[:, None], node_indices] = adj_ 
            z_global[:, node_indices, :] = z
            zsc_global[:, node_indices, :] = z_scaled
            eps_global[:, node_indices, :] = eps
            start_idx += train_nodes[i]

        return adj_global, mu, logvar, z_global, zsc_global, eps_global, rk, snr


class GraphDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, zdim, dropout=0.0, gdc='bp'):
        super(GraphDecoder, self).__init__()
        self.dropout = dropout
        self.gdc = gdc
        self.zdim = zdim
        self.rk_lgt = torch.nn.Parameter(torch.FloatTensor(torch.Size([1, zdim])))
        self.reset_parameters()
        self.SMALL = 1e-16
        #self.adjdrpt = adjdrpt
        #self.edge_drop = DropAdj(self.adjdrpt)

    def reset_parameters(self):
        torch.nn.init.uniform_(self.rk_lgt, a=-6., b=0.)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        assert self.zdim == z.shape[2], 'zdim not compatible!'

        # The variable 'rk' in the code is the square root of the same notation in
        # http://proceedings.mlr.press/v80/yin18b/yin18b-supp.pdf
        # i.e., instead of do Z*diag(rk)*Z', we perform [Z*diag(rk)] * [Z*diag(rk)]'.
        rk = torch.sigmoid(self.rk_lgt).pow(.5)

        # Z shape: [J, N, zdim]
        # Z' shape: [J, zdim, N]
        z = z.mul(rk.view(1, 1, self.zdim))
        adj_lgt = torch.bmm(z, torch.transpose(z, 1, 2))

        # 1 - exp( - exp(ZZ'))
        adj_lgt = torch.clamp(adj_lgt, min=-np.Inf, max=25)
        adj = 1 - torch.exp(-adj_lgt.exp())

        # if not self.training:
        #     adj = torch.mean(adj, dim=0, keepdim=True)
        
        return adj, z, rk.pow(2)


# Source Code Adapted from PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html#VGAE

class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder.
    """

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        sigmoid: bool = True,
    ) -> torch.Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        return torch.sigmoid(value) if sigmoid else value


    def forward_all(self, z: torch.Tensor, sigmoid: bool = True) -> torch.Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
    

class MLPDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=16, num_layers=2,
                 dropout=0.0):
        super(MLPDecoder, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z, edge_index, sigmoid: bool = True):
        x = z[edge_index[0]] * z[edge_index[1]]
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = (self.lins[-1](x)).sum(dim=1)
        return torch.sigmoid(x) if sigmoid else x


class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None
                ):
        super().__init__()

        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> torch.Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> torch.Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor,
                   neg_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: torch.Tensor, pos_edge_index: torch.Tensor,
             neg_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
        from eval import evaluate_mrr

        pos_edge_index = pos_edge_index.t()
        if neg_edge_index.dim() >= 3: neg_edge_index = torch.permute(neg_edge_index, (2, 0, 1)).view(2, -1)
        else: neg_edge_index = neg_edge_index.t()

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)

        pos_mrr_pred = torch.flatten(pos_pred)
        neg_mrr_pred = neg_pred.squeeze(-1)
        
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred), pos_mrr_pred, neg_mrr_pred

    def test_full_batch(self, loader, device, model) -> torch.Tensor:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

        y_pred, y_true = [], []
        for data in loader:
            data = data.to(device)
            
            src, tgt = data.edge_label_index

            z_o = torch.zeros(data.num_nodes, device=device)

            z_o[src[data.edge_label == 1]] = 1
            z_o[tgt[data.edge_label == 1]] = 1
            
            z = model.encode(data.x, data.adj_t, z_o.long())

            pred = self.decoder(z, data.edge_label_index, sigmoid=True)
            
            y_pred.append(pred.view(-1))
            y_true.append(data.edge_label.view(-1).to(torch.float))
        vgae_pred, vgae_y = torch.cat(y_pred), torch.cat(y_true)

        pos_pred = vgae_pred[vgae_y==1]
        neg_pred = vgae_pred[vgae_y==0]

        y, pred =  vgae_y.detach().cpu().numpy(), vgae_pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred), pos_pred, neg_pred


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`. If set to
            :obj:`None`, will default to the
            :class:`gnn_model.GCN_Encoder`.
            (default: :obj:`None`)
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)
        
    def reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> torch.Tensor:
        """"""  # noqa: D419
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z
    
    def forward_all(self, *args, **kwargs):
        return self.decoder.forward_all(*args, **kwargs)


    def kl_loss(self, mu: Optional[torch.Tensor] = None,
                logstd: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """

        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
  
class GCN_Encoder(torch.nn.Module):
    '''
    Deterministic GCN Encoder class for VGAE and GAE, with conditional option for SEAL Subgraph labels
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.0, edge_drop=0.0):
        super(GCN_Encoder, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.max_z = 2
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.adjdrop = DropAdj(edge_drop)

        print("in channels", in_channels)
        print("hidden channels", hidden_channels)
        print("Self.convs: ", self.convs)
        print("Self lins: ", self.lins)

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, 2 * out_channels))
            self.lins.append(LayerNorm(2 * out_channels))

        elif num_layers > 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.lins.append(LayerNorm(hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels))
                self.lins.append(LayerNorm(hidden_channels))
            self.convs.append(GCNConv(hidden_channels, 2 * out_channels))

        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

        self.dropout = dropout
        
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     
    def forward(self, x, adj_t, z=None, ablate=False):
        adj_t_drop = self.adjdrop(adj_t)

        if self.invest == 1:
            print('layers in gcn_encoder: ', len(self.convs))
            self.invest = 0
        
        if not ablate:
            z_emb = self.z_embedding(z)
            x = torch.cat([x, z_emb.to(torch.float)], dim=1)
            
        for conv, lin in zip(self.convs[:-1], self.lins):
            x = conv(x, adj_t_drop)
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t_drop)
        x = F.relu(x)
        
        return self.conv_mu(x, adj_t_drop)