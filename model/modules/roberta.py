# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
from typing import Optional, List

import torch
from torch import nn
from transformers import AutoConfig, RobertaModel, RobertaConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class ExtendedRoberta(nn.Module):
    def __init__(self, max_input_length, model="roberta-base"):
        super().__init__()
        config: RobertaConfig = AutoConfig.from_pretrained(model)
        config.max_position_embeddings = max_input_length + 2
        self.roberta: RobertaModel = RobertaModel.from_pretrained(model, config=config, ignore_mismatched_sizes=True)
        # huggingface randomly initializes parameter weights, if we increase the max input length
        # with the following lines we ensure to keep the learned weights
        original_model = RobertaModel.from_pretrained(model)
        self.roberta.embeddings.position_embeddings.weight.data[:514] = original_model.embeddings.position_embeddings.weight.data

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.forward
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask,
                               past_key_values=past_key_values,
                               use_cache=use_cache,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict)

        if isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions):
            return outputs.last_hidden_state
        else:
            return outputs[0]

    def __str__(self):
        return str(self.roberta)
