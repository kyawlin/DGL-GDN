import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
    ):
        super(GATLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = nn.Linear(
            self.in_channels, self.heads * self.out_channels, bias=False
        )

        self.attn_fc = nn.Linear(2 * self.out_channels, 1, bias=False)
        self.attn_embed = nn.Linear(2 * self.out_channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_embed.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        embed = torch.cat([edges.src["embed"], edges.dst["embed"]], dim=1)
        a = self.attn_fc(z2) + self.attn_embed(embed)

        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, graph, h, embedding):

        with graph.local_scope():
            z = self.lin(h)

            graph.ndata["z"] = z
            graph.ndata["embed"] = embedding

            graph.apply_edges(self.edge_attention)

            graph.update_all(self.message_func, self.reduce_func)
            return graph.ndata.pop("h")

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge="cat", bias=False):
        super(MultiHeadGATLayer, self).__init__()

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

        if bias is not None:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, graph, h, embedding):

        with graph.local_scope():

            head_outs = [attn_head(graph, h, embedding) for attn_head in self.heads]

            if self.merge == "cat":

                # concat on the output feature dimension (dim=1)
                rst = torch.cat(head_outs, dim=1)
            else:
                # merge using average
                rst = torch.mean(torch.stack(head_outs))

            if self.bias is not None:
                return rst + self.bias
            else:
                return rst
