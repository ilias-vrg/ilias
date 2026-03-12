import torch
import copy
from typing import Optional
import torch.nn.functional as F
from torch import nn, Tensor


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(m.weight, std=0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.zeros_(m.bias)


class AMES(nn.Module):
    def __init__(
        self,
        model_dim=128,
        nhead=2,
        num_encoder_layers=5,
        dim_feedforward=1024,
        activation="relu",
        normalize_before=False,
        binarized=True,
    ):

        super(AMES, self).__init__()
        self.binarized = binarized

        self.mtc_token = nn.Parameter(torch.rand(model_dim))
        if self.binarized:
            self.remap_local = nn.Sequential(
                nn.Linear(model_dim, model_dim), nn.LayerNorm(model_dim)
            )
        else:
            self.remap_local = nn.Identity()

        decoder_layer = TransformerDecoderLayer(
            model_dim, nhead, dim_feedforward, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(model_dim) if normalize_before else None
        self.decoder = TransformerDecoder(
            decoder_layer, num_encoder_layers, decoder_norm
        )
        self.classifier = nn.Linear(model_dim, 1)
        nn.init.trunc_normal_(self.mtc_token, std=0.02)
        self.apply(_init_weights)

    def forward(
        self,
        src_local=None,
        src_mask=None,
        tgt_local=None,
        tgt_mask=None,
        return_logits=False,
    ):

        src_local = self.remap_local(src_local)
        tgt_local = self.remap_local(tgt_local)

        B, Q, D = src_local.size()
        B, T, D = tgt_local.size()

        mtc_token = self.mtc_token[None, None, :].repeat(B, 1, 1)

        if not self.training and not self.binarized:
            tgt_local = tgt_local.half().float()

        input_feats = torch.cat([mtc_token, src_local, tgt_local], 1).permute(1, 0, 2)

        src_mask = (
            src_local.new_zeros((B, Q), dtype=torch.bool)
            if src_mask is None
            else src_mask
        )
        tgt_mask = (
            tgt_local.new_zeros((B, T), dtype=torch.bool)
            if tgt_mask is None
            else tgt_mask
        )

        input_mask = torch.cat(
            [src_local.new_zeros((B, 1), dtype=torch.bool), src_mask, tgt_mask], 1
        ).bool()

        sa_mask = src_local.new_zeros((Q + T + 1, Q + T + 1), dtype=torch.bool)
        sa_mask[1 : Q + 1, Q + 1 :] = 1.0
        sa_mask[Q + 1 :, 1 : Q + 1] = 1.0

        ca_mask = src_local.new_zeros((Q + T + 1, Q + T + 1), dtype=torch.bool)
        ca_mask[1 : Q + 1, 1 : Q + 1] = 1.0
        ca_mask[Q + 1 :, Q + 1 :] = 1.0

        logits = self.decoder(
            input_feats,
            sa_mask=sa_mask,
            ca_mask=ca_mask,
            src_key_padding_mask=input_mask,
        )
        sim = self.classifier(logits[0]).view(-1)

        if return_logits:
            return sim, logits.permute(1, 0, 2)
        return sim


class TransformerDecoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        sa_mask: Optional[Tensor] = None,
        ca_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):

        output = src.permute(1, 0, 2)
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                sa_mask=sa_mask,
                ca_mask=ca_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        output = output.permute(1, 0, 2)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, activation, normalize_before):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(
            d_model, 2 * dim_feedforward if activation == "geglu" else dim_feedforward
        )
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask
        )[0]
        return x

    def _ca_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.cross_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask
        )[0]
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x

    def forward(
        self,
        x,
        sa_mask: Optional[Tensor] = None,
        ca_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            x = x + self._sa_block(self.norm1(x), sa_mask, src_key_padding_mask)
            x = x + self._ca_block(self.norm2(x), ca_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, sa_mask, src_key_padding_mask))
            x = self.norm2(x + self._ca_block(x, ca_mask, src_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
