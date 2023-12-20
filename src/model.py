import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import metrics
import math
import copy
from torch import einsum
from dataset import PT_FEATURE_SIZE
from basic_blocks import SetBlock, BasicConv1d
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from GCN import GCN
CHAR_SMI_SET_LEN = 64

class CovarianceCalculator(nn.Module):
    def forward(self, x):
        B, C, H, W = x.size()
        embedding_vectors = x.view(B, C, H, W)
        cov = torch.zeros(B, C, H, H).to(embedding_vectors.device)
        for i in range(B):
            cov[i, :, :, :] = self.torch_cov(embedding_vectors[i, :, :, :])
        return cov

    def torch_cov(self, X):
        D = X.shape[-1]
        mean = torch.mean(X, dim=-1).unsqueeze(-1)
        X = X - mean
        out = 1 / (D - 1) * X @ X.transpose(-1, -2)
        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output


class DilatedParllelResidualBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
#             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):

        # reduce
        output1 = self.c1(input)
        # output1 = output1.type(torch.float32)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        # combine = combine.type(torch.float32)
        output = self.br2(combine)
        return output

class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
#             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        # merge
        combine = torch.cat([d1, add1, add2, add3], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output

class MFTLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(MFTLayer, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.decoder = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None,
    tgt_key_padding_mask=None, memory_key_padding_mask=None):
        e_out = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        t_out = self.decoder(tgt, e_out, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask)

        return e_out, t_out

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, qkv):
        b, n, _ = qkv.shape
        h = self.heads
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

device = "cuda" if torch.cuda.is_available() else "cpu"

class PfgPDI(nn.Module):

    def __init__(self, max_seq_len, max_smi_len):
        super().__init__()

        smi_embed_size = 128
        seq_embed_size = 128
        
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN, smi_embed_size)
        self.smiPE = PositionalEncoding(smi_embed_size, 0.1, max_smi_len)

        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out})
        self.proPE = PositionalEncoding(seq_embed_size, 0.1, max_seq_len)

        nhead = 8
        dim_feedforward = 1024
        num_layers = 4
        mft_layer = MFTLayer(smi_embed_size, nhead, dim_feedforward)
        self.layers = _get_clones(mft_layer, num_layers)
        self.e_norm = nn.LayerNorm(smi_embed_size)
        self.d_norm = nn.LayerNorm(smi_embed_size)

        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedParllelResidualBlockA(ic, oc))
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool1d(1))  # (N, oc)
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)
        self.gcn_seq = GCN(in_fea=1000, num_nodes=128, out_fea=512)
        # ks = {32: 1, 64: 5, seq_oc: 3}
        # for oc in [32, seq_oc]:
        #     conv_seq.append(SetBlock(BasicConv1d(ic, oc, ks[oc]), pooling=True))
        #     ic = oc
        # conv_seq.append(nn.AdaptiveMaxPool1d(1))
        # conv_seq.append(Squeeze())
        # self.conv_seq = nn.Sequential(
        #     SetBlock(BasicConv1d(ic, seq_oc, 3), pooling=True),
        #     nn.AdaptiveMaxPool1d(1),
        #     Squeeze()
        # )

        # (N, H=32, L)
        conv_pkt = []
        ic = seq_embed_size
        for oc in [32, 64, pkt_oc]:
            conv_pkt.append(nn.Conv1d(ic, oc, 3))  # (N,C,L)
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        # conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        # conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)  # (N,oc)

        # ks = {32: 1, 64: 3, pkt_oc: 3}
        # for oc in [32, pkt_oc]:
        #     conv_pkt.append(SetBlock(BasicConv1d(ic, oc, ks[oc]), pooling=True))
        #     ic = oc
        # conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        # conv_pkt.append(Squeeze())
        # self.conv_pkt = nn.Sequential(
        #     SetBlock(BasicConv1d(ic, pkt_oc, 3), pooling=True),
        #     nn.AdaptiveMaxPool1d(1),
        #     Squeeze()
        # )
        self.cov = CovarianceCalculator()

        conv_smi = []
        ic = smi_embed_size
        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
            ic = oc
        # conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,128)
        self.gcn_smi = GCN(in_fea=150, num_nodes=128, out_fea=1)
        # ks = {32: 1, 64: 3, smi_oc: 3}
        # ps = {32: 1, 64: 3, smi_oc: False}
        # for oc in [32, smi_oc]:
        #     conv_smi.append(SetBlock(BasicConv1d(ic, oc, ks[oc]), pooling=ps[oc]))
        #     ic = oc
        # conv_smi.append(nn.AdaptiveMaxPool1d(1))
        # conv_smi.append(Squeeze())
        # self.conv_smi = nn.Sequential(
        #     SetBlock(BasicConv1d(ic, smi_oc, 3), pooling=True),
        #     nn.AdaptiveMaxPool1d(1),
        #     Squeeze()
        # )

        self.gcn_all = GCN(in_fea=1024, num_nodes=128, out_fea=1)
        self.cat_dropout = nn.Dropout(0.1)

        self.classifier = nn.Sequential(
            nn.Linear(seq_oc+smi_oc+128, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1),
            # nn.PReLU()
        )
        

    def forward(self, seq, pkt, smi, proMask, smiMask, tgt_mask):
        # assert seq.shape == (N,L,43)
        seq_embed = self.seq_embed(seq)  # (N,L,32)
        src = self.proPE(seq_embed)
        src = src.permute(1, 0, 2)

        smi_embed = self.smi_embed(smi)  # (N,L,32)
        target = self.smiPE(smi_embed)
        target = target.permute(1, 0, 2)

        src_key_padding_mask = ~(proMask.to(torch.bool))
        tgt_key_padding_mask = ~(smiMask.to(torch.bool))
        memory_key_padding_mask = ~(proMask.to(torch.bool))

        e_out = src
        d_out = target
        tgt_mask = tgt_mask.squeeze(0)
        for idx, mod in enumerate(self.layers):
            e_out, d_out = mod(e_out, d_out, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, \
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
            e_out, d_out = mod(e_out, d_out)

            # e_out = self.e_norm(e_out)
            # d_out = self.d_norm(d_out)

        smi_embed = d_out.permute(1, 0, 2) + smi_embed
        # seq_embed = e_out.permute(1, 0, 2)

        seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,32,L)
        seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert pkt.shape == (N,L,43)
        pkt_embed = self.seq_embed(pkt)  # (N,L,32)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)  # (N,128)

        # assert smi.shape == (N, L)
        # smi_embed = self.smi_embed(smi)  # (N,L,32)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

        pkt_covariance = self.cov(pkt_conv.unsqueeze(1)).squeeze()
        # seq_conv = seq_conv.transpose(-1, -2)
        # smi_conv = smi_conv.transpose(-1, -2)

        smi_gcn = self.gcn_smi(smi_conv, pkt_covariance).flatten(1)

        pkt_pool = pkt_conv.max(-1)[0]
        if seq_conv.ndim == 1:
            seq_conv = seq_conv.unsqueeze(0)
        all_emd = torch.cat([seq_conv, pkt_pool, smi_gcn], dim=-1)
        # all_emd = self.gcn_all(all_emd, pkt_covariance).flatten(1)

        # cat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=-1)  # (N,128*3)

        output = self.classifier(all_emd)
        return output


def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    total_sample = 0
    outputs = []
    targets = []
    T = nn.Transformer()

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for idx, (*x, y) in pbar:
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)
            total_sample += len(y)

            seq, pkt, smi, proMask, smiMask = x
            tgt_mask = T.generate_square_subsequent_mask(smi.shape[1]).tolist()
            tgt_mask = [tgt_mask] * 1
            tgt_mask = torch.as_tensor(tgt_mask).to(device)

            y_hat = model(seq, pkt, smi, proMask, smiMask, tgt_mask)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).cpu().numpy()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

            pbar.set_description(f' *--* Loss={test_loss / total_sample:.3f}')

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation, outputs
