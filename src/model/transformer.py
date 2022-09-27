# https://github.com/facebookresearch/detr

import copy
import torch
import torch.nn as nn

  
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        d_ff: int, 
        dropout: float=0.1
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor
    ) -> torch.Tensor:
        query = key = x + p
        value = x
        a, _ = self.attention(query, key, value)
        b = self.norm_1(x+self.dropout_1(a))
        o = self.norm_2(b+self.dropout_2(self.ffn(b)))
        return o
    
    
class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        encoder_layer: TransformerEncoderLayer, 
        num_layers: int
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(
        self, 
        x: torch.Tensor,
        p: torch.Tensor
    ) -> torch.Tensor:
        output = x
        for layer in self.layers:
            output = layer(output, p)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        d_ff: int, 
        dropout: float=0.1
    ) -> None:
        super().__init__()
        self.attention_1 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attention_2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout=dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        query = key = y + q
        value = y        
        a, _ = self.attention_1(query, key, value)
        b = self.norm_1(y+self.dropout_1(a))
        
        query = b + q
        key = x + p
        value = x
        c, _ = self.attention_2(query, key, value)
        d = self.norm_2(b+self.dropout_2(c))        
        o = self.norm_3(d+self.dropout_3(self.ffn(d)))
        return o
    
    
class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer: TransformerDecoderLayer, 
        num_layers: int
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(
        self, 
        x: torch.Tensor,
        p: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        output = y
        for layer in self.layers:
            output = layer(x, p, output, q)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout=0.1
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self, 
        d_model: int=512,
        n_head: int=8, 
        num_encoder_layers: int=6,
        num_decoder_layers: int=6, 
        d_ff: int=2048, 
        dropout: float=0.1
    ) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            n_head=n_head, 
            d_ff=d_ff, 
            dropout=dropout
        )
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_encoder_layers
        )

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, 
            n_head=n_head,
            d_ff=d_ff,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer, 
            num_layers=num_decoder_layers
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, 
        x: torch.Tensor,
        p: torch.Tensor,
        y: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """
        x: Height x Width, Batch, N_hidden
        p: Height x Width, Batch, N_hidden
        y: Sequence, Batch, N_hidden
        q: Sequence, Batch, N_hidden
        out: Batch, Sequcne, N_hidden
        """
        memory = self.encoder(x, p)
        out = self.decoder(memory, p, y, q)
        return memory.permute(1, 2, 0), out.permute(1, 2, 0)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])