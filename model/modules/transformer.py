# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
import torch
from torch import nn
from torch.nn import functional

from model.options import TransformerOptions


class Embeddings(nn.Module):
    def __init__(self, n_vocabulary, d_output, padding_idx):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(n_vocabulary, d_output, padding_idx=padding_idx)
        sqrt_d_output = torch.sqrt(torch.as_tensor(d_output))
        self.register_buffer("sqrt_d_output", sqrt_d_output)

    def forward(self, x):
        return self.embedding(x) * self.sqrt_d_output


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_output, dropout_rate, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # copied from: The Annotated Transformer (v2022)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_output)
        position = torch.arange(0, max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_output, 2) * -(math.log(10000.0) / d_output))
        div_term = 1 / torch.pow(10000, torch.arange(0, d_output, 2) / d_output)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_output: int, d_query, d_key, d_value, dropout_rate, n_heads, d_head=None, d_query_key_output=None):
        """
        Initialization of a multi-headed attention module, where intermediate feature dimensions, in particular the
        feature dimension of a single head output, can be set individually.
                :param d_output: the feature dimension of the module output
                :param d_query: the feature dimension of the queries
                :param d_key: the feature dimension of the keys
                :param d_value: the feature dimension of the values
                :param d_query_key_output: the feature dimension of the projection of queries and keys
                :param d_head: the feature dimension of a single head output
                :param n_heads: the number of heads
        """
        super(MultiHeadedAttention, self).__init__()
        d_query_key_output = torch.as_tensor(d_query_key_output if d_query_key_output else int(d_output / n_heads))
        self.register_buffer("d_query_key_output", d_query_key_output)
        d_head = torch.as_tensor(d_head if d_head else int(d_output / n_heads))
        self.register_buffer("d_head", d_head)
        scale_factor: torch.Tensor = torch.sqrt(torch.as_tensor(self.d_query_key_output))
        self.register_buffer("scale_factor", scale_factor)
        n_heads = torch.as_tensor(n_heads)
        self.register_buffer("n_heads", n_heads)

        self.query_projection = nn.Linear(d_query, self.d_query_key_output * self.n_heads)
        self.key_projection = nn.Linear(d_key, self.d_query_key_output * self.n_heads)
        self.value_projection = nn.Linear(d_value, self.d_head * self.n_heads)

        self.dropout = nn.Dropout(dropout_rate)
        self.multi_headed_projection = nn.Linear(self.d_head * self.n_heads, d_output)

    def forward(self, input_q, input_k, input_v, mask=None, attention_propagation=None):
        """
        :param input_q: size: (batch_dimensions) x (query input length) x (query-key dimension)
        :param input_k: size: (batch_dimensions) x (key-value input length) x (query-key dimension)
        :param input_v: size: (batch_dimensions) x (key-value input length) x (value dimension)
        :param mask: (batch_dimensions) x (query input length) x (key-value input length)
        :param attention_propagation:
        :return:
        """
        q = self.query_projection(input_q)
        k = self.key_projection(input_k)
        v = self.value_projection(input_v)

        q = torch.stack(q.split(self.d_query_key_output, -1), -3)
        k = torch.stack(k.split(self.d_query_key_output, -1), -3)
        v = torch.stack(v.split(self.d_head, -1), -3)

        qk = torch.matmul(q, k.transpose(-2, -1)) / self.scale_factor

        attention_weights = qk
        if attention_propagation is not None:
            attention_weights.masked_fill_((1 - mask).bool(), 0)
            attention_weights = torch.matmul(attention_weights, attention_propagation.transpose(-1, -2))

        if mask is not None:
            attention_weights.masked_fill_((1 - mask).bool(), -float('inf'))

        attention_weights = functional.softmax(attention_weights, -1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0)

        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        output = torch.cat(output.split(1, -3), -1).squeeze(-3)
        output = self.multi_headed_projection(output)
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_input, d_hidden_size, d_output, dropout_rate):
        """
        Initializes a two-layer feed forward network with a ReLu activation function in between.
                :param d_input:
                :param d_hidden_size:
                :param d_output:
        """
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(d_input, d_hidden_size)
        self.layer2 = nn.Linear(d_hidden_size, d_output)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        output = self.layer1(input)
        output = self.dropout(functional.relu(output))
        output = self.layer2(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, d_output, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(d_output)

    def forward(self, residual, layer_output):
        return self.norm(residual + self.dropout(layer_output))


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step for pytorch's CrossEntropyLoss.
    We do not apply softmax, as this is done inside CrossEntropyLoss.
    """

    def __init__(self, d_input, n_vocabulary):
        super(Generator, self).__init__()
        self.projection = nn.Linear(d_input, n_vocabulary)

    def forward(self, x):
        return self.projection(x)  # we do not take the log_softmax of the projection because we will train using the cross entropy loss


class StandardEncoderLayer(nn.Module):
    def __init__(self, d_input, d_intermediate, d_output, d_ffn, dropout_rate, n_heads):
        super(StandardEncoderLayer, self).__init__()
        self.self_attention: MultiHeadedAttention \
            = MultiHeadedAttention(d_intermediate, d_input, d_input, d_input, dropout_rate, n_heads)
        self.ffn: FeedForwardNetwork \
            = FeedForwardNetwork(d_intermediate, d_ffn if d_ffn else d_intermediate * 4, d_output, dropout_rate)
        self.residual_block1 = ResidualBlock(d_intermediate, dropout_rate)
        self.residual_block2 = ResidualBlock(d_output, dropout_rate)

    def forward(self, input, mask=None):
        intermediate = self.residual_block1(input, self.self_attention(input, input, input, mask=mask))
        return self.residual_block2(intermediate, self.ffn(intermediate))


class StandardEncoder(nn.Module):
    def __init__(self, n_layers, options: TransformerOptions):
        super(StandardEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            StandardEncoderLayer(options.hidden_size, options.hidden_size, options.hidden_size,
                                 options.ffn_inner_hidden_size, options.dropout_rate, options.num_attention_heads)
            for _ in range(0, n_layers)
        ])

    def forward(self, inputs, self_attention_mask=None):
        for layer in self.encoder_layers:
            inputs = layer(inputs, mask=self_attention_mask)

        return inputs


class StandardDecoderLayer(nn.Module):
    def __init__(self, d_input, d_causal_self_attention, d_cross_attention, d_output, d_ffn, dropout_rate, n_heads):
        super(StandardDecoderLayer, self).__init__()
        self.causal_self_attention: MultiHeadedAttention = MultiHeadedAttention(d_causal_self_attention,
                                                                                d_input,
                                                                                d_input,
                                                                                d_input,
                                                                                dropout_rate,
                                                                                n_heads)

        self.cross_attention: MultiHeadedAttention = MultiHeadedAttention(d_cross_attention,
                                                                          d_causal_self_attention,
                                                                          d_causal_self_attention,
                                                                          d_causal_self_attention,
                                                                          dropout_rate,
                                                                          n_heads)

        self.ffn: FeedForwardNetwork = FeedForwardNetwork(d_cross_attention, d_ffn, d_output, dropout_rate)
        self.residual_block1 = ResidualBlock(d_causal_self_attention, dropout_rate)
        self.residual_block2 = ResidualBlock(d_cross_attention, dropout_rate)
        self.residual_block3 = ResidualBlock(d_output, dropout_rate)

    def forward(self, input, encoder_output, self_attention_mask=None, encoder_mask=None):
        intermediate = self.residual_block1(input,
                                            self.causal_self_attention(input, input, input, mask=self_attention_mask))
        intermediate = self.residual_block2(intermediate,
                                            self.cross_attention(intermediate, encoder_output, encoder_output,
                                                                 mask=encoder_mask))
        return self.residual_block3(intermediate, self.ffn(intermediate))


class StandardDecoder(nn.Module):
    def __init__(self, n_layers, options: TransformerOptions):
        super(StandardDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList([
            StandardDecoderLayer(options.hidden_size, options.hidden_size, options.hidden_size, options.hidden_size,
                                 options.ffn_inner_hidden_size, options.dropout_rate, options.num_attention_heads)
            for _ in range(0, n_layers)])

    def forward(self, x, encoder_output, self_attention_mask=None, encoder_mask=None):
        output = x
        for layer in self.decoder_layers:
            output = layer(output, encoder_output, self_attention_mask=self_attention_mask, encoder_mask=encoder_mask)
        return output
