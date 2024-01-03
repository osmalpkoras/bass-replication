# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
import math

import torch

from model.models import Bass


def beam_search(model: Bass, batch, options, beam_width=5, length_penalty=0.9, max_decoding_length=math.inf, min_decoding_length=0, bos_id=0, pad_id=1, eos_id=2):
    device = options.device
    # method argument initialization
    max_decoding_length = int(min(max_decoding_length, options.transformer_options.max_input_length))
    length_penalty = torch.as_tensor(length_penalty, device=device)
    bos_id = torch.as_tensor(bos_id, device=device)
    pad_id = torch.as_tensor(pad_id, device=device)
    eos_id = torch.as_tensor(eos_id, device=device)

    # initialization for first prediction, starting with a bos token
    encoder_input_ids, encoder_padding, decoder_input_ids, decoder_padding, graph_construction_matrix, graph_padding, adjacency_matrix = (i.to(device) for i in batch)
    batch_size = encoder_input_ids.size(0)
    decoder_input_ids = torch.ones(batch_size, 1, device=device, dtype=encoder_input_ids.dtype) * bos_id
    decoder_padding = torch.ones(decoder_input_ids.size(), dtype=encoder_padding.dtype, device=device)

    # Make first predictions for this batch
    _, output, _ = model(encoder_input_ids, encoder_padding, decoder_input_ids, decoder_padding, graph_construction_matrix, graph_padding, adjacency_matrix, include_tokens=torch.ones(batch_size, 1) * beam_width)
    results = torch.topk(torch.nn.functional.log_softmax(output, dim=-1), dim=-1, k=beam_width)
    probs = results.values
    tokens = results.indices

    # create beam_width many branches for each batch item
    beam_batch_size = batch_size * beam_width
    new_order = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, beam_width).view(-1)
    decoder_input_ids = torch.cat([decoder_input_ids[new_order], torch.reshape(tokens, (-1, 1))], dim=1)
    decoder_padding = torch.ones(decoder_input_ids.size(), dtype=decoder_padding.dtype, device=device)
    pad = torch.ones(beam_batch_size, 1, dtype=decoder_input_ids.dtype, device=device) * pad_id
    scores = torch.reshape(probs, (-1, 1))
    incomplete_sequences = decoder_input_ids[:, -1] != eos_id
    encoder_input_ids = encoder_input_ids[new_order, :]
    encoder_padding = encoder_padding[new_order, :]
    graph_construction_matrix = graph_construction_matrix[new_order, :]
    graph_padding = graph_padding[new_order, :]
    adjacency_matrix = adjacency_matrix[new_order, :]

    sequence_ids = torch.arange(0, beam_batch_size, device=device)
    best_finished_beam_score = [None for _ in range(0, batch_size)]
    trigram_dictionary = [{} for _ in range(0, beam_batch_size)]
    for decoding_length in range(2, max_decoding_length):
        if not torch.any(incomplete_sequences):
            break
        beam_batches = torch.split(incomplete_sequences, beam_width)
        incomplete_decoder_input_ids = decoder_input_ids[incomplete_sequences,:]
        _, output, _ = model(encoder_input_ids[incomplete_sequences,:], encoder_padding[incomplete_sequences,:], incomplete_decoder_input_ids, decoder_padding[incomplete_sequences,:], graph_construction_matrix[incomplete_sequences,:,:], graph_padding[incomplete_sequences,:], adjacency_matrix[incomplete_sequences,:,:], include_tokens=torch.ones(int(torch.sum(incomplete_sequences)), 1) * beam_width)

        if decoding_length < min_decoding_length:
            output[:, eos_id] = -float("inf")

        # block token-trigrams
        if decoding_length > 2:
            for k in sequence_ids[incomplete_sequences]:
                blocked_tokens = trigram_dictionary[k].setdefault(tuple(decoder_input_ids[k, -3:-1].tolist()), [])
                blocked_tokens.append(decoder_input_ids[k, -1])

                blocked_tokens = trigram_dictionary[k].get(tuple(decoder_input_ids[k, -2:].tolist()))
                if blocked_tokens:
                    output[torch.sum(incomplete_sequences[:k+1]) - 1, blocked_tokens] = -float("inf")

        # get results
        vocab_size = output.size(-1)
        log_probs = torch.nn.functional.log_softmax(output, dim=-1)
        log_probs += scores[incomplete_sequences,:]

        # think of how to vectorize this! this might be possible if you reshape incomplete_sequences or so
        splits = [torch.sum(c) for c in beam_batches]
        log_probs_split = torch.split(log_probs[:, :], splits, dim=0)
        decoder_input_ids_split = torch.split(decoder_input_ids[incomplete_sequences, :], splits, dim=0)
        select_indices = torch.arange(0, beam_batch_size, device=incomplete_sequences.device)
        new_tokens = []
        new_scores = []
        for j in range(0, batch_size):
            remaining_beam_size = int(torch.sum(beam_batches[j]))
            if remaining_beam_size == 0:
                continue
            p = torch.reshape(log_probs_split[j], (-1, 1))
            results = torch.topk(p, dim=0, k=remaining_beam_size)

            # Resolve beam origin and true word ids.
            topk_beam_index = results.indices.div(vocab_size).to(torch.long).view(-1)
            topk_ids = results.indices.fmod(vocab_size)

            subindices = torch.zeros(incomplete_sequences.size(), dtype=torch.bool, device=incomplete_sequences.device)
            subindices[j*beam_width:(j+1)*beam_width] = incomplete_sequences[j*beam_width:(j+1)*beam_width]
            select_indices[subindices] = topk_beam_index + j*beam_width

            # Append last prediction.
            t = torch.cat(
                [decoder_input_ids_split[j][topk_beam_index, :],
                topk_ids], -1)

            new_scores.append(results.values)
            new_tokens.append(t)

            newly_finished_beam_indices = new_tokens[-1][:,-1] == eos_id
            penalized_new_scores = new_scores[-1] / torch.pow((5 + decoding_length) / 6, length_penalty)
            if torch.any(newly_finished_beam_indices):
                penalized_new_finished_scores = penalized_new_scores[newly_finished_beam_indices]
                if best_finished_beam_score[j] is None or torch.any(penalized_new_finished_scores > best_finished_beam_score[j]):
                    best_finished_beam_score[j] = torch.max(penalized_new_finished_scores)

            if best_finished_beam_score[j]:
                new_tokens[-1][(penalized_new_scores < best_finished_beam_score[j]).view(-1), -1] = eos_id

        scores[incomplete_sequences] = torch.cat(new_scores, dim=0)
        decoder_input_ids = torch.cat([decoder_input_ids, pad], dim=1)
        decoder_input_ids[incomplete_sequences, :] =  torch.cat(new_tokens, dim=0)
        decoder_padding = (decoder_input_ids != pad_id).to(decoder_padding.dtype)
        incomplete_sequences = torch.logical_and(incomplete_sequences, decoder_input_ids[:, -1] != eos_id)

        trigram_dictionary = [trigram_dictionary[ind].copy() for ind in select_indices]

    # apply length penalty to scores
    lengths = torch.ones(scores.size(), dtype=decoder_input_ids.dtype, device=device)
    lengths[:] = max_decoding_length
    l = (decoder_input_ids == eos_id).nonzero(as_tuple=True)
    lengths[l[0], 0] = l[1]
    penalized_scores = scores / torch.pow((5 + lengths) / 6, length_penalty)
    indices = torch.max(penalized_scores.reshape((batch_size, beam_width)), dim=-1).indices
    # select best summary
    generated_summaries = decoder_input_ids.reshape((batch_size, beam_width, -1))[torch.arange(0, batch_size), indices, :]
    return generated_summaries
