# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 01:06:51 2022

@author: Jinuk Kwon
"""

from module_conformer import *


class Conformer_Mel(nn.Module):
    # d_model : number of features
    def __init__(self, d_input, d_output, d_model=512, nhead=16, e_layers=3, d_layers=3, d_bi=False, conv_dropout=0.5,
                 ff_dropout=0.5, conv_expansion_factor=2,
                 kernel_size=31, ff_expansion_factor=4,
                 device='cuda'):
        super(Conformer_Mel, self).__init__()
        self.src_add_emb = Linear(d_input, 1).to(device)
        self.d_layers = d_layers
        self.drop = nn.Dropout(0.1)
        self.emb_in = Linear(d_input, d_model)

        self.conformer_encoder = ConformerEncoder(d_model=d_model, nlayers=e_layers, nheads=nhead,
                                                  feed_forward_expansion_factor=ff_expansion_factor,
                                                  kernel_size=kernel_size, conv_dropout_p=conv_dropout,
                                                  feed_forward_dropout_p=ff_dropout,
                                                  conv_expansion_factor=conv_expansion_factor).to(device)

        self.emb_out = Linear(d_output, d_model).to(device)
        if not (d_layers == 0):
            self.rnn = nn.LSTM(d_model, d_model, d_layers, batch_first=True, bidirectional=d_bi)
        self.out = Linear(d_model * (d_bi + 1), d_output).to(device)

    def forward(self, src, src_mask=None):
        # src_mask
        src_mask = src_mask[:, :, :src.size(1)]
        encoding = self.emb_in(src)
        encoding = self.conformer_encoder(encoding, src_mask)
        output = encoding
        if not (self.d_layers == 0):
            output, _ = self.rnn(output)
        output = self.out(output)
        return output


class ConformerEncoder(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            nlayers: int = 17,
            nheads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            kernel_size: int = 31,
            ff_residu_factor=0.5,
    ):
        super(ConformerEncoder, self).__init__()

        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=d_model,
            num_attention_heads=nheads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=kernel_size,
            ff_residu_factor=ff_residu_factor,
        ) for _ in range(nlayers)])

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs, mask=None):
        # outputs = self.input_projection(inputs)
        outputs = inputs

        for layer in self.layers:
            outputs = layer(outputs, mask)

        return outputs


class ConformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            ff_residu_factor=0.5,
    ):
        super(ConformerBlock, self).__init__()
        self.feed_forward_residual_factor = ff_residu_factor

        self.ff1 = FeedForwardModule(encoder_dim=encoder_dim, expansion_factor=feed_forward_expansion_factor,
                                     dropout_p=feed_forward_dropout_p, )

        self.attn = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p)

        self.conv_block = ConformerConvModule(
            in_channels=encoder_dim,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p)

        self.ff2 = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p)
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) * self.feed_forward_residual_factor + x
        x = self.attn(x, mask=mask) + x
        x = self.conv_block(x) + x
        x = self.ff2(x) * self.feed_forward_residual_factor + x
        x = self.layer_norm(x)
        return x
