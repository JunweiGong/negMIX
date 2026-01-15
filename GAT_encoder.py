import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 out_dim,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads, dropout=attn_drop, negative_slope=negative_slope))

        # hidden layers
        for l in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden * heads, num_hidden, heads, dropout=attn_drop, negative_slope=negative_slope))

        self.linear_layer = nn.Linear(num_layers * heads * num_hidden, out_dim)

    def forward(self, x, edge_index):
        h = x
        layer_output = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.feat_drop, training=self.training)
            h = self.gat_layers[l](h, edge_index)
            h = F.elu(h)
            layer_output.append(h)

        views = torch.stack(layer_output)
        final_emb = torch.cat(layer_output, dim=1)
        logits = self.linear_layer(final_emb)
        output_prob = F.softmax(logits, dim=1)

        return views, final_emb, logits, output_prob

