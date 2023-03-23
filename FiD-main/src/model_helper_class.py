# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements modified
- transformers.T5Stack
    - Output cross-attention weights
- transformers.T5ForConditionalGeneration
    - Initialize it with modified T5Stack
"""

import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np


class T5StackNew(transformers.T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

    # forward of T5StackNew is modified so that it outputs cross-attention weights
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_value_states=None,
            use_cache=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            if self.is_decoder:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if past_key_value_states is not None:
            assert seq_length == 1, "Input shape is {}, but should be {} when using past_key_value_sates".format(
                input_shape, (batch_size, 1)
            )
            # required mask seq length can be calculated via length of past
            # key value states and seq_length = 1 for the last token
            mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length
        else:
            mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length).to(inputs_embeds.device)

        # initialize past_key_value_states with `None` if past does not exist
        if past_key_value_states is None:
            past_key_value_states = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, self.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = ()
        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if self.output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[4 if self.output_attentions else 3]
            # append next layer key value states
            present_key_value_states = present_key_value_states + (present_key_value_state,)

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)
            outputs = outputs + (present_key_value_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (presents,) (all hidden states), (all attentions)


# modify the decoder such that it is from class T5StackNew
class T5ForConditionalGenerationNew(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5StackNew(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()
