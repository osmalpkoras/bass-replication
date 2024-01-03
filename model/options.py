# Paper Title: A Second Look on BASS - Boosting Abstractive Summarization with Unified Semantic Graphs
# Author: Osman Alperen Koras
# Affiliation: Institute for AI in Medicine (IKIM), University Medicine Essen
# Email: osman.koras@uni-due.de
# Date: 12.10.2023
#
# License: MIT
class TransformerOptions:
    def __init__(self):
        self.max_input_length = 512

        self.hidden_size = 512
        self.num_attention_heads = 8
        self.ffn_inner_hidden_size = self.hidden_size * 4
        self.dropout_rate = 0.1


class OptimizerOptions:
    def __init__(self):
        self.beta1 = 0.9
        self.beta2 = 0.998
        self.learning_rate = None
        self.warmup_steps = None


class Options:
    def __init__(self):
        super().__init__()
        # architectural configuration
        self.pretrained_language_model = "roberta-base"

        self.transformer_options: TransformerOptions = TransformerOptions()
        self.transformer_options.max_input_length = 1024
        self.transformer_options.hidden_size = 768
        self.transformer_options.num_attention_heads = 8
        self.transformer_options.ffn_inner_hidden_size = 2048
        self.transformer_options.dropout_rate = 0.1

        self.num_graph_encoder_layers = 2
        self.num_decoder_layers = 6

        self.teleport_probability = 0.9
        self.propagation_steps = 2

        # training options
        self.label_smoothing_factor = 0.1
        self.maximum_gradient_norm = 0.2

        # optimizer configuration
        self.optimizer_for_pretrained_part = OptimizerOptions()
        self.optimizer_for_pretrained_part.learning_rate = 0.002
        self.optimizer_for_pretrained_part.warmup_steps = 20000

        self.optimizer_for_other_parts = OptimizerOptions()
        self.optimizer_for_other_parts.learning_rate = 0.1
        self.optimizer_for_other_parts.warmup_steps = 10000

        # decoding configuration
        self.beam_size = 5
        self.length_penalty_factor = 0.9
