# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
import torch
from torch import nn

from model.modules.bass import BassDecoder, cross_attention_masks
from model.modules.roberta import ExtendedRoberta
from model.modules.transformer import Embeddings, PositionalEncoding, StandardEncoder, Generator, StandardDecoder
from model.options import Options


class Bass(nn.Module):
    def __init__(self, options: Options):
        super(Bass, self).__init__()
        self.options: Options = options
        max_input_length = self.options.transformer_options.max_input_length
        d_input = self.options.transformer_options.hidden_size
        self.text_encoder: ExtendedRoberta = ExtendedRoberta(max_input_length, self.options.pretrained_language_model)

        vocab_size = self.text_encoder.roberta.config.vocab_size
        self.decoder_embedding = nn.Sequential(Embeddings(vocab_size, d_input, self.text_encoder.roberta.config.pad_token_id),
                                               PositionalEncoding(d_input, self.options.transformer_options.dropout_rate, max_len=max_input_length))
        self.decoder_embedding[0].embedding.weight = self.text_encoder.roberta.embeddings.word_embeddings.weight

        self.graph_encoder = StandardEncoder(self.options.num_graph_encoder_layers, self.options.transformer_options)
        self.decoder = BassDecoder(self.options.num_decoder_layers, self.options.transformer_options)

        self.generator = Generator(d_input, vocab_size)
        self.generator.projection.weight = self.decoder_embedding[0].embedding.weight

        causal_self_attention_mask = torch.tril(torch.ones([max_input_length, max_input_length], dtype=torch.long, requires_grad=False))
        causal_self_attention_mask = causal_self_attention_mask.unsqueeze(-3).unsqueeze(-3)
        self.register_buffer("causal_self_attention_mask", causal_self_attention_mask)

    def get_propagation_matrix(self, adjacency_matrix: torch.Tensor):
        degrees = torch.sum(adjacency_matrix != 0, -1, keepdim=True)
        degrees[degrees == 0] = 1
        degree_normalized_adjacency_matrix = adjacency_matrix / degrees

        degree_normalized_adjacency_matrix_potency = torch.eye(degree_normalized_adjacency_matrix.size(-1), device=adjacency_matrix.device).repeat(adjacency_matrix.size(0), 1, 1, 1)
        teleport_probability_potency = torch.as_tensor(1., device=adjacency_matrix.device)
        sum = teleport_probability_potency * degree_normalized_adjacency_matrix_potency
        teleport_probability = self.options.teleport_probability
        for i in range(1, self.options.propagation_steps):
            degree_normalized_adjacency_matrix_potency *= degree_normalized_adjacency_matrix
            teleport_probability_potency *= teleport_probability
            sum += teleport_probability_potency * degree_normalized_adjacency_matrix_potency

        sum *= (1 - teleport_probability)
        degree_normalized_adjacency_matrix_potency *= degree_normalized_adjacency_matrix
        teleport_probability_potency *= teleport_probability
        sum += teleport_probability_potency * degree_normalized_adjacency_matrix_potency

        return sum

    def forward(self, encoder_input_ids, encoder_padding, decoder_input_ids, decoder_padding, graph_construction_matrix, graph_padding, adjacency_matrix, include_tokens=None):
        # load graph information for document if available
        # else: create graph information and save
        # graph = None  # size is (batch size) x (input length) x (feature dimension)
        # adjacency_matrix = None  # size is (batch size) x (input length) x (input length)
        # attention_propagation_matrix = None  # size is (batch size) x (input length) x (input length)

        adjacency_matrix, text_encoder_output, graph_encoder_output = self.encode(encoder_input_ids, encoder_padding, graph_construction_matrix, adjacency_matrix)
        output = self.decode(encoder_padding, decoder_input_ids, decoder_padding, graph_padding, adjacency_matrix, text_encoder_output, graph_encoder_output)

        output = self.generator(output)
        if include_tokens is not None:
            if include_tokens[0] == 1:
                results = torch.max(torch.nn.functional.log_softmax(output, dim=-1), -1)
                probs = results.values
                tokens = results.indices
                return None, probs, tokens
            else:
                # results = torch.topk(torch.nn.functional.log_softmax(output[:,-1,:], dim=-1), dim=-1, k=int(self.options.transformer_options.max_input_length / 3))
                # probs = results.values
                # tokens = results.indices
                return None, output[:,-1,:], None
        else:
            target = decoder_input_ids.detach().clone()
            target[:, :-1] = decoder_input_ids[:, 1:]
            target[:, -1] = self.text_encoder.roberta.config.pad_token_id
            # target = decoder_input_ids.detach().clone()
            # target[:, :-1] = decoder_input_ids[:, 1:]
            # target[:, -1] = -1
            # torch.masked_fill(target, decoder_padding == 0, -1)
            loss = torch.nn.functional.cross_entropy(output.permute(0,2,1), target, ignore_index=self.text_encoder.roberta.config.pad_token_id, label_smoothing=self.options.label_smoothing_factor, reduction="mean")
            return loss.unsqueeze(0), None, None

    def decode(self, encoder_padding, decoder_input_ids, decoder_padding, graph_padding, adjacency_matrix, text_encoder_output, graph_encoder_output):
        causal_self_attention_mask = self.causal_self_attention_mask[:, :,
                                                                     :decoder_input_ids.size(-1),
                                                                     :decoder_input_ids.size(-1)]

        graph_cross_attention_mask = cross_attention_masks(decoder_padding, graph_padding)
        text_cross_attention_mask = cross_attention_masks(decoder_padding, encoder_padding)

        embedded_decoder_input = self.decoder_embedding(decoder_input_ids)
        output = self.decoder(embedded_decoder_input, graph_encoder_output, text_encoder_output,
                              self_attention_mask=torch.mul(causal_self_attention_mask, cross_attention_masks(decoder_padding, decoder_padding)),
                              graph_encoder_mask=graph_cross_attention_mask,
                              text_encoder_mask=text_cross_attention_mask,
                              graph_propagation=self.get_propagation_matrix(adjacency_matrix))

        return output

    def encode(self, encoder_input_ids, encoder_padding, graph_construction_matrix, adjacency_matrix):
        text_encoder_output = self.text_encoder(encoder_input_ids, attention_mask=encoder_padding)

        degrees = torch.sum(graph_construction_matrix != 0, -1, keepdim=True)
        degrees[degrees == 0] = 1
        graph_input = torch.matmul(graph_construction_matrix, text_encoder_output) / degrees
        adjacency_matrix = adjacency_matrix.unsqueeze(-3)
        graph_encoder_output = self.graph_encoder(graph_input, self_attention_mask=adjacency_matrix)
        return adjacency_matrix, text_encoder_output, graph_encoder_output


class RTS2S(nn.Module):
    def __init__(self, options: Options):
        super(RTS2S, self).__init__()
        self.options: Options = options
        max_input_length = self.options.transformer_options.max_input_length
        d_input = self.options.transformer_options.hidden_size
        self.text_encoder: ExtendedRoberta = ExtendedRoberta(max_input_length, self.options.pretrained_language_model)

        vocab_size = self.text_encoder.roberta.config.vocab_size
        self.decoder_embedding = nn.Sequential(
            Embeddings(vocab_size, d_input, self.text_encoder.roberta.config.pad_token_id),
            PositionalEncoding(d_input, self.options.transformer_options.dropout_rate, max_len=max_input_length))
        self.decoder_embedding[0].embedding.weight = self.text_encoder.roberta.embeddings.word_embeddings.weight
        self.decoder = StandardDecoder(self.options.num_decoder_layers, self.options.transformer_options)

        self.generator = Generator(d_input, vocab_size)
        self.generator.projection.weight = self.decoder_embedding[0].embedding.weight

        causal_self_attention_mask = torch.tril(
            torch.ones([max_input_length, max_input_length], dtype=torch.long, requires_grad=False))
        causal_self_attention_mask = causal_self_attention_mask.unsqueeze(-3).unsqueeze(-3)
        self.register_buffer("causal_self_attention_mask", causal_self_attention_mask)


    def forward(self, encoder_input_ids, encoder_padding, decoder_input_ids, decoder_padding, include_tokens=None):
        encoder_output, second_encoder_output = self.encode(encoder_input_ids, encoder_padding)
        output = self.decode(encoder_padding, decoder_input_ids, decoder_padding, encoder_output)

        output = self.generator(output)
        if include_tokens is not None:
            if include_tokens[0] == 1:
                results = torch.max(torch.nn.functional.log_softmax(output, dim=-1), -1)
                probs = results.values
                tokens = results.indices
                return None, probs, tokens
            else:
                return None, output[:, -1, :], None
        else:
            target = decoder_input_ids.detach().clone()
            target[:, :-1] = decoder_input_ids[:, 1:]
            target[:, -1] = self.text_encoder.roberta.config.pad_token_id
            loss = torch.nn.functional.cross_entropy(output.permute(0, 2, 1), target, ignore_index=self.text_encoder.roberta.config.pad_token_id, label_smoothing=self.options.label_smoothing_factor, reduction="mean")

            return loss.unsqueeze(0), None, None

    def decode(self, encoder_padding, decoder_input_ids, decoder_padding, encoder_output):
        causal_self_attention_mask = self.causal_self_attention_mask[:, :, :decoder_input_ids.size(-1), :decoder_input_ids.size(-1)]
        cross_attention_mask = cross_attention_masks(decoder_padding, encoder_padding)
        embedded_decoder_input = self.decoder_embedding(decoder_input_ids)

        output = self.decoder(embedded_decoder_input, encoder_output,
                              self_attention_mask=torch.mul(causal_self_attention_mask, cross_attention_masks(decoder_padding, decoder_padding)),
                              encoder_mask=cross_attention_mask)
        return output

    def encode(self, encoder_input_ids, encoder_padding):
        encoder_output = self.text_encoder(encoder_input_ids, attention_mask=encoder_padding)
        return encoder_output


class exRTS2S(nn.Module):
    def __init__(self, options: Options):
        super(exRTS2S, self).__init__()
        self.options: Options = options
        max_input_length = self.options.transformer_options.max_input_length
        d_input = self.options.transformer_options.hidden_size
        self.text_encoder: ExtendedRoberta = ExtendedRoberta(max_input_length, self.options.pretrained_language_model)

        vocab_size = self.text_encoder.roberta.config.vocab_size
        self.decoder_embedding = nn.Sequential(
            Embeddings(vocab_size, d_input, self.text_encoder.roberta.config.pad_token_id),
            PositionalEncoding(d_input, self.options.transformer_options.dropout_rate, max_len=max_input_length))
        self.decoder_embedding[0].embedding.weight = self.text_encoder.roberta.embeddings.word_embeddings.weight

        self.second_text_encoder = StandardEncoder(self.options.num_graph_encoder_layers,
                                                   self.options.transformer_options)
        self.decoder = BassDecoder(self.options.num_decoder_layers, self.options.transformer_options)

        self.generator = Generator(d_input, vocab_size)
        self.generator.projection.weight = self.decoder_embedding[0].embedding.weight

        causal_self_attention_mask = torch.tril(
            torch.ones([max_input_length, max_input_length], dtype=torch.long, requires_grad=False))
        causal_self_attention_mask = causal_self_attention_mask.unsqueeze(-3).unsqueeze(-3)
        self.register_buffer("causal_self_attention_mask", causal_self_attention_mask)

    def forward(self, encoder_input_ids, encoder_padding, decoder_input_ids, decoder_padding, include_tokens=None):
        encoder_output, second_encoder_output = self.encode(encoder_input_ids, encoder_padding)
        output = self.decode(encoder_padding, decoder_input_ids, decoder_padding, encoder_output, second_encoder_output)

        output = self.generator(output)
        if include_tokens is not None:
            if include_tokens[0] == 1:
                results = torch.max(torch.nn.functional.log_softmax(output, dim=-1), -1)
                probs = results.values
                tokens = results.indices
                return None, probs, tokens
            else:
                return None, output[:, -1, :], None
        else:
            target = decoder_input_ids.detach().clone()
            target[:, :-1] = decoder_input_ids[:, 1:]
            target[:, -1] = self.text_encoder.roberta.config.pad_token_id
            loss = torch.nn.functional.cross_entropy(output.permute(0, 2, 1), target, ignore_index=self.text_encoder.roberta.config.pad_token_id, label_smoothing=self.options.label_smoothing_factor, reduction="mean")

            return loss.unsqueeze(0), None, None

    def decode(self, encoder_padding, decoder_input_ids, decoder_padding, encoder_output, second_encoder_output=None):
        causal_self_attention_mask = self.causal_self_attention_mask[:, :, :decoder_input_ids.size(-1), :decoder_input_ids.size(-1)]
        cross_attention_mask = cross_attention_masks(decoder_padding, encoder_padding)
        embedded_decoder_input = self.decoder_embedding(decoder_input_ids)

        output = self.decoder(embedded_decoder_input,
                              second_encoder_output, encoder_output,
                              self_attention_mask=torch.mul(causal_self_attention_mask, cross_attention_masks(decoder_padding, decoder_padding)),
                              graph_encoder_mask=cross_attention_mask,
                              text_encoder_mask=cross_attention_mask)

        return output

    def encode(self, encoder_input_ids, encoder_padding):
        encoder_output = self.text_encoder(encoder_input_ids, attention_mask=encoder_padding)
        second_encoder_output = self.second_text_encoder(encoder_output, self_attention_mask=cross_attention_masks(encoder_padding, encoder_padding))
        return encoder_output, second_encoder_output
