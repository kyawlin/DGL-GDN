import math

import torch
import dgl
from dgl.data import DGLDataset
import torch.nn as nn
import torch.nn.functional as F
from attentionLayer import MultiHeadGATLayer


class GDN(nn.Module):
    def __init__(
        self,
        graph,
        embed_dim=64,
        input_dim=10,
        hidden_dim=128,
        outlayer_hidden_dim=128,
        num_outlayers=1,
        topk=20,
        dropout=0.2,
        device="cpu",
    ):

        super(GDN, self).__init__()

        self.graph = graph

        node_num = self.graph.num_nodes()

        self.embedding = nn.Embedding(node_num, embed_dim)

        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        self.multiHeadGAT = MultiHeadGATLayer(
            input_dim, embed_dim, num_heads=1, merge="cat", bias=True
        )

        self.dp = nn.Dropout(dropout)
        self.topk = topk
        modules = []

        self.outlayer = nn.Linear(embed_dim, 1)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

        nn.init.xavier_normal_(self.outlayer.weight, gain=gain)

    def forward(self, data):

        x = data.clone().detach()

        B, N, Ft = x.shape

        x = x.view(-1, Ft).contiguous()

        all_embeddings = self.embedding(torch.arange(N).to(self.device))

        weights_arr = all_embeddings.detach().clone()

        all_embeddings = all_embeddings.repeat(B, 1)

        weights = weights_arr.view(N, -1)

        cos_ji_mat = torch.matmul(weights, weights.T)

        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1)
        )
        cos_ji_mat = cos_ji_mat / normed_mat

        dim = weights.shape[-1]
        topk_num = self.topk

        topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

        self.learned_graph = topk_indices_ji

        gated_i = (
            torch.arange(0, N).repeat(1, topk_num).flatten().long()  # nodenum  # topk
        )
        gated_j = topk_indices_ji.flatten().long()

        gated_edge_index = torch.cat(
            (gated_j.unsqueeze(0), gated_i.unsqueeze(0)), dim=0
        )
        batch_gated_edge_index = get_batch_edge_index(gated_edge_index, B, N).to(
            self.device
        )
        gated_graph = dgl.graph((batch_gated_edge_index[0], batch_gated_edge_index[1]))

        gat_out = self.multiHeadGAT(gated_graph, x, embedding=all_embeddings)

        gat_out = gat_out.view(B, N, -1)

        indx = torch.arange(0, N).to(self.device)
        rst = torch.mul(gat_out, self.embedding(indx))

        rst = rst.permute(0, 2, 1)

        rst = F.relu(self.bn_outlayer_in(rst))
        rst = rst.permute(0, 2, 1)

        rst = self.dp(rst)

        rst = self.outlayer(rst)

        return rst.view(-1, N)


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num : (i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()
