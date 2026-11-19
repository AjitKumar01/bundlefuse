"""RaMen as described in the ECAI 2025 paper (doi:10.3233/FAIA251022, Sec. 3), built on the released code.

The released code (github.com/Rec4Fun/RaMen, commit b96038b) is used for everything that matches the paper. These
places differ from the paper, or leave a part of the model switched off, and are changed here:

  1. Bundle encoder, eq. (4). models/utils.py:162 keeps the attention layers in a plain Python list, so their
     projections Psi_B^K, Psi_B^Q are never registered and never trained (they stay at their random start). Here they
     are an nn.ModuleList. L2 = number of these layers.
  2. Item encoder, eq. (3). The released code applies one attention layer; the paper stacks L1 layers (shared
     Psi_I^K, Psi_I^Q, as written in eq. 3).
  3. Multi-modal fusion, eq. (2). Psi_rho is a linear map R^{2d} -> R^d; the released code uses a two-layer MLP.
  4. Collaborative encoder, eq. (6): c_i = beta * mean_{n=1..N} s_i^(n) + (1 - beta) * s_i. The released code averages
     the input together with the N layer outputs and then adds the input again with weight 1 (residual_type 0).
  5. Explicit fusion, eqs. (7)-(8): g = gamma * p + (1 - gamma) * c, one gamma for items and bundles. The released
     defaults (item_alpha = bundle_alpha = 1) give the collaborative encoder weight 0.
  6. Hypergraph, eq. (11): Z layers. The released code fixes Z = 1.
  7. Gumbel-softmax, eq. (10). The released code also samples the noise at test time, so two scorings of the same
     query differ (by up to 0.77 in score on Electronic). Here the noise is used in training only; at test the
     assignment is softmax(f / tau).
  8. Alignment loss, eqs. (14), (17): L = L_NLL + lambda_1 (L_ICL + L_BCL). The released code weights the two terms
     0.1 and 0.05.

Unchanged from the paper: d = 64, lr 1e-3, batch 1024, L2 weight 1e-5, tau = 0.2, item-item graph E = X^T X with
edge if E_ij >= eps (eps = 1 for e-commerce), inner-product score (15), softmax NLL (16). Tuned, as in Sec. 4.1.3:
L1, L2, N, Z in {1..5}; H in {4, 8, 16, 32, 64}; beta, gamma, lambda_1 in {0.1, ..., 0.9}.

Kept from the released code although the paper does not mention them: separate encoder (bundle side) and decoder
(item side) networks, as in CLHE; dropout 0.7 on the hyperedge assignments during training; dropout 0.1 on the
graph attention weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import models.RaMen as R
from models.utils import TransformerEncoder


class PaperAmatrix(R.Amatrix):
    """Eq. (6), inner part: mean over the N layer outputs s^(1..N), without the input s^(0)."""

    def forward(self, x, edge_index, return_attention_weights=True):
        feats = []
        for conv in self.convs:
            x, _ = conv(x, edge_index, return_attention_weights=True)
            feats.append(x)
        return torch.stack(feats, 1).mean(1), None


class PaperEncoder(R.HierachicalEncoder):
    def __init__(self, conf, raw_graph, features):
        super().__init__(conf, raw_graph, features)
        self.L1 = conf['L1'];self.beta = conf['beta'];self.n_hyper_layer = conf['Z']
        self.hgnnLayer = R.HGNNLayer(self.n_hyper_layer)
        self.mlp_multimodal = nn.Linear(2*self.embedding_size, self.embedding_size)  # Psi_rho, eq. (2)
        R.init(self.mlp_multimodal)
        self.iui_gat_conv = PaperAmatrix(in_dim=64, out_dim=64, n_layer=conf['n_layer_gat'], dropout=0.1, heads=self.n_head,
                                         concat=False, self_loop=False, extra_layer=True)
        for unused in ('w_v', 'image_trs', 'text_trs', 'image_embedding', 'text_embedding'):  # never used in a forward pass
            if hasattr(self, unused):delattr(self, unused)

    def selfAttention(self, features):
        """Eq. (3) applied L1 times with shared Psi_I^K, Psi_I^Q, then mean pooling over the two rows of rho_i."""
        for _ in range(self.L1):
            attn = (self.w_q(features).mul(self.embedding_size**-0.5) @ self.w_k(features).transpose(-1, -2)).softmax(-1)
            features = attn @ features
        return features.mean(-2)

    def assign(self, f):
        """Eq. (10): Gumbel-softmax while training, plain softmax(f / tau) at test."""
        return F.gumbel_softmax(f, self.tau, dim=1, hard=False) if self.training else (f/self.tau).softmax(1)

    def collaborative(self):
        """Eq. (6): c_i, and c_b as the mean of c_i over the bundle's items."""
        s = self.iui_gat_conv(self.items_embedding, self.iui_edge_index)[0]
        c_item = self.beta*s + (1 - self.beta)*self.items_embedding
        return c_item, self.bundle_agg_graph_ori @ c_item

    def hypergraph(self, c_item, c_bundle, c_feature, t_feature):
        """Eqs. (9)-(12) for both modalities, summed and L2-normalised."""
        Y = self.scipy_sparse_to_torch_sparse(self.bi_graph).to(self.device);out = 0
        for feat, W in ((c_feature, self.v_hyper), (t_feature, self.t_hyper)):
            fi = feat @ W;fb = torch.mm(Y, fi)
            b, i = self.hgnnLayer(self.drop(self.assign(fi)), self.drop(self.assign(fb)), c_item, c_bundle)
            out = out + torch.cat([b, i], 0)
        return torch.split(F.normalize(out), [self.num_bundle, self.num_item], 0)

    def characteristic(self, c_feature, t_feature):
        """Eqs. (1)-(3): p_i for every item."""
        mm = self.mlp_multimodal(torch.cat([c_feature, t_feature], 1))
        rho = torch.stack([mm, self.item_embeddings], -2)
        return self.selfAttention(F.normalize(rho, dim=-1))

    def forward(self, seq, all=False, test=False, discrete=False):
        c_item, c_bundle = self.collaborative()
        c_feature, t_feature = self.c_encoder(self.content_feature), self.t_encoder(self.text_feature)
        phi_bundle, phi_item = self.hypergraph(c_item, c_bundle, c_feature, t_feature)
        p_item = self.characteristic(c_feature, t_feature)
        if all:  # item side: g_i, eq. (7)
            return self.item_alpha*p_item + (1 - self.item_alpha)*c_item, phi_item, c_item
        seq = seq.masked_fill(seq == self.num_item, 0)
        return p_item[seq], phi_bundle, c_bundle, c_item, phi_item


def build(conf, graphs, features):
    """RAMEN with the paper's parts: PaperEncoder on both sides and a registered L2-layer bundle encoder (eq. 4)."""
    conf = dict(conf, item_alpha=conf['gamma'], bundle_alpha=conf['gamma'], alpha_item_loss=conf['lambda1'],
                alpha_bundle_loss=conf['lambda1'], trans_layer=conf['L2'])
    model = models.RAMEN(conf, graphs, features)
    model.encoder, model.decoder = PaperEncoder(conf, graphs, features), PaperEncoder(conf, graphs, features)
    model.bundle_encode.attn_encode = nn.ModuleList(model.bundle_encode.attn_encode)
    return model
