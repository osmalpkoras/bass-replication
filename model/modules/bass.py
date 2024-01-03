# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
import torch
from torch import nn

from model.modules.transformer import MultiHeadedAttention, FeedForwardNetwork, ResidualBlock
from model.options import TransformerOptions


class BassDecoderLayer(nn.Module):
    def __init__(self, d_input, d_causal_self_attention, d_graph_cross_attention, d_text_cross_attention, d_fusion,
                 d_output, d_ffn, dropout_rate, n_heads):
        super(BassDecoderLayer, self).__init__()
        self.causal_self_attention: MultiHeadedAttention = MultiHeadedAttention(d_causal_self_attention,
                                                                                d_input,
                                                                                d_input,
                                                                                d_input, 
                                                                                dropout_rate,
                                                                                n_heads)
        
        self.graph_cross_attention: MultiHeadedAttention = MultiHeadedAttention(d_graph_cross_attention,
                                                                                d_causal_self_attention,
                                                                                d_causal_self_attention,
                                                                                d_causal_self_attention, 
                                                                                dropout_rate,
                                                                                n_heads)

        self.text_cross_attention: MultiHeadedAttention = MultiHeadedAttention(d_text_cross_attention,
                                                                               d_causal_self_attention,
                                                                               d_causal_self_attention,
                                                                               d_causal_self_attention,
                                                                               dropout_rate,
                                                                               n_heads)
        
        self.graph_text_projection = nn.Linear(d_graph_cross_attention + d_text_cross_attention, d_fusion)
        self.ffn: FeedForwardNetwork = FeedForwardNetwork(d_fusion, d_ffn, d_output, dropout_rate)
        self.residual_block1 = ResidualBlock(d_causal_self_attention, dropout_rate)
        self.residual_block2 = ResidualBlock(d_fusion, dropout_rate)
        self.residual_block3 = ResidualBlock(d_output, dropout_rate)

    def forward(self, input, graph_encoder_output, text_encoder_output, self_attention_mask=None,
                graph_encoder_mask=None, text_encoder_mask=None, graph_propagation=None):
        intermediate = self.residual_block1(input, self.causal_self_attention(input, input, input, mask=self_attention_mask))

        graph_cross_attention_output = self.graph_cross_attention(intermediate, graph_encoder_output, graph_encoder_output, mask=graph_encoder_mask, attention_propagation=graph_propagation)
        text_cross_attention_output = self.text_cross_attention(intermediate, text_encoder_output, text_encoder_output, mask=text_encoder_mask)
        fusion_output = self.graph_text_projection(torch.cat([graph_cross_attention_output, text_cross_attention_output], -1))

        intermediate = self.residual_block2(intermediate, fusion_output)
        return self.residual_block3(intermediate, self.ffn(intermediate))


class BassDecoder(nn.Module):
    def __init__(self, n_layers, options: TransformerOptions):
        super(BassDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList([
            BassDecoderLayer(options.hidden_size, options.hidden_size, options.hidden_size, options.hidden_size, options.hidden_size, options.hidden_size, options.ffn_inner_hidden_size, options.dropout_rate, options.num_attention_heads)
            for _ in range(0, n_layers)])

    def forward(self, x, graph_encoder_output, text_encoder_output, self_attention_mask=None,
                graph_encoder_mask=None, text_encoder_mask=None, graph_propagation=None):
        output = x
        for layer in self.decoder_layers:
            output = layer(output, graph_encoder_output, text_encoder_output, self_attention_mask=self_attention_mask,
                           graph_encoder_mask=graph_encoder_mask, text_encoder_mask=text_encoder_mask,
                           graph_propagation=graph_propagation)
        return output


def cross_attention_masks(left_attention_mask: torch.Tensor, right_attention_mask: torch.Tensor):
    return torch.matmul(left_attention_mask.unsqueeze(-2).unsqueeze(-1), right_attention_mask.unsqueeze(-2).unsqueeze(-2))