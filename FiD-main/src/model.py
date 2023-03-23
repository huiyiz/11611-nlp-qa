# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np

# from .model_helper_class import T5ForConditionalGenerationNew


# use modified T5ForConditionalGenerationNew from model_helper_class
# class FiDT5(transformers.T5ForConditionalGeneration):
class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)



class ProbGeneration(nn.Module): 
    def __init__(self, seq_len, d): 
        super(ProbGeneration, self).__init__()
        self.linear1 = nn.Linear(d, 1) # encoder embedding [seq_len, d] -> [seq_len,1]
        self.linear2 = nn.Linear(d, 1) # decoder layer repr [5, d]      -> [5,1]
        self.sigmoid = nn.Sigmoid()
        print("ProbGeneration: matrix W shape ", seq_len, d)

    def forward(self, x, y): 
        print("ProbGeneration: input x shape ", x.size())
        print("ProbGeneration: input y shape ", y.size())
        out = self.sigmoid(self.linear1(x) + self.linear2(y))
        return out



class FiDPGN(FiDT5):
    def __init__(self, config):
        super().__init__(config)
        # self.wrap_encoder()
        # embedding dimension: (config.vocab_size, config.d_model)
        self.prob_gen = ProbGeneration(config.vocab_size, config.d_model)
        self.decoder.output_attentions = True
        print("config.vocab_size ", config.vocab_size, ";  config.d_model ", config.d_model)

    def forward(self, input_ids=None, attention_mask=None, **kwargs): 
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        # prepare for parameters: 
        # input_ids, attention_mask
        # encoder_outputs, decoder_input_ids, decoder_attention_mask, decoder_past_key_value_states, use_cache, 
        # labels, inputs_embeds, decoder_inputs_embeds, head_mask, output_attentions, output_hidden_states
        encoder_outputs = kwargs.get("encoder_outputs", None)
        decoder_input_ids = kwargs.get("decoder_input_ids", None)
        decoder_attention_mask = kwargs.get("decoder_attention_mask", None)
        decoder_past_key_value_states = kwargs.get("decoder_past_key_value_states", None)
        use_cache = kwargs.get("use_cache", None)
        labels = kwargs.get("labels", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        decoder_inputs_embeds = kwargs.get("decoder_inputs_embeds", None)
        head_mask = kwargs.get("head_mask", None)
        output_attentions = kwargs.get("output_attentions", None)
        output_hidden_states = kwargs.get("output_hidden_states", None)

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        print("decoder_input_ids: ", decoder_input_ids)
        print("decoder_inputs_embeds: ", decoder_inputs_embeds)
        print("decoder_past_key_value_states: ", decoder_past_key_value_states)

        # Decode: last-layer hidden state, (present_key_value_states,) (all hidden states), (all attentions)
        # self.decoder.output_attentions = True
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            # output_attentions=output_attentions,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
        )

        # check decoder outputs
        print("attention_mask shape", attention_mask.size())
        print("use_cache: ", use_cache,
              "\toutput_hidden_states: ", output_hidden_states,
              "\toutput_attentions: ", output_attentions,
              )
        print("decoder_outputs[0]", decoder_outputs[0])
        print("PRINTING SHAPE OF decoder_outputs", len(decoder_outputs))
        print_output(decoder_outputs)


        print("input_id shape", input_ids.size())

        # insert decoder past at right place
        # to speed up decoding
        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0] # [1, 5, 768]
        print("sequence_output: ", sequence_output.size())
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output) # self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        print("lm_logits: ", lm_logits.size()) # [1, 5, vocab_size]

        # <PGN structure>

        # mod = self.decoder.block[-1]
        # attn = mod.layer[1].EncDecAttention

        # self.encoder is EncoderWrapper object, need to do .encoder to get type T5Stack object
        input_embedding_token = self.encoder.encoder.embed_tokens(input_ids) # [1, 25000, 768]
        output_repr_decoder = decoder_outputs[0] # [1, 5, 768]
        print("input_embedding_token: ", input_embedding_token.size())
        print("output_repr_decoder: ", output_repr_decoder.size())

        print("lm_logits: ", lm_logits.size())

        # p_gen = self.prob_gen.forward(input_embedding_token, output_repr_decoder)
        # print("p_gen", p_gen)

        decoder_attn = self.get_cross_attention_weights(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=True,
        )
        print("decoder_attn: ", decoder_attn.size())
        copy_dist = self.get_crossattention_scores(attention_mask)
        print("copy_dist: ", copy_dist.size())
        dist = torch.add(p_gen*lm_logits + (1-p_gen)*copy_dist)

        # replace lm_logits with dist in the following code
        lm_logits = dist
        # </PGN structure>
        
        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            decoder_outputs = (loss,) + decoder_outputs

        # decoder_outputs: loss, lm_logits, (present_key_value_states,) (all hidden states), (all attentions)
        # encoder_outputs: last-layer hidden state, (present_key_value_states,) (all hidden states), (all attentions)

        # Return values:
        # loss: 
        #       Classification loss (cross entropy)
        # prediction_scores: Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). 
        #       If past_key_value_states is used only the last prediction_scores of the sequences of shape (batch_size, 1, hidden_size) is output.
        # decoder_past_key_value_states: 
        #       Contains pre-computed key and value hidden-states of the attention blocks
        # hidden_states: 
        #       Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) 
        #       Shape (batch_size, sequence_length, hidden_size).
        # attentions: Tuple of torch.FloatTensor (one for each layer) 
        #       Shape (batch_size, num_heads, sequence_length, sequence_length)
        #       Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        return decoder_outputs + encoder_outputs


    def get_cross_attention_weights(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            past_key_value_states=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            use_cache=None,
            output_attentions=True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            assert self.decoder.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.decoder.embed_tokens(input_ids)

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
        if encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length).to(inputs_embeds.device)

        # initialize past_key_value_states with `None` if past does not exist
        if past_key_value_states is None:
            past_key_value_states = [None] * len(self.decoder.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.decoder.get_extended_attention_mask(attention_mask, input_shape, self.decoder.device)

        if encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.decoder.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.decoder.get_head_mask(head_mask, self.decoder.config.num_layers)
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.decoder.dropout(inputs_embeds)

        for i, (layer_module, past_key_value_state) in enumerate(zip(self.decoder.block, past_key_value_states)):
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
                print("OUTPUT ATTENTIONS: ", self.decoder.output_attentions)
                print("length of layer outputs: ", len(layer_outputs))
                print(type(layer_outputs[0]), type(layer_outputs[1]), type(layer_outputs[2]), type(layer_outputs[3]))
                print(layer_outputs[0].size(), len(layer_outputs[1]), layer_outputs[2].size(), layer_outputs[3].size())
                # print("\nlayer outputs[0]")
                # print(layer_outputs[0])
                # print("\nlayer outputs[1]")
                # print(layer_outputs[1])
                # print("\nlayer outputs[2]")
                # print(layer_outputs[2])
                # print("\nlayer outputs[3]")
                # print(layer_outputs[3])
                position_bias = layer_outputs[3 if self.decoder.output_attentions else 2]
                position_bias = layer_outputs[2]
                if encoder_hidden_states is not None:
                    # encoder_decoder_position_bias = layer_outputs[4 if self.decoder.output_attentions else 3]
                    encoder_decoder_position_bias = layer_outputs[3]
            else:
                print("length of layer outputs: ", len(layer_outputs))
                print(type(layer_outputs[0]), type(layer_outputs[1]))
                print(layer_outputs[0], layer_outputs[1])
            # all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
            last_layer_attention = layer_outputs[-1]
        # return all_attentions  # (all attentions)
        return last_layer_attention


def print_output(output, level=0):
    if isinstance(output, tuple):
        st = "\t" * level + "tuple of length " + str(len(output))
        print(st)
        for i in output:
            print_output(i, level+1)
    else:
        st = "\t" * level + "tensor: "
        print(st, output.size())




class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        return outputs

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output

class RetrieverConfig(transformers.BertConfig):

    def __init__(self,
                 indexing_dimension=768,
                 apply_question_mask=False,
                 apply_passage_mask=False,
                 extract_cls=False,
                 passage_maxlength=200,
                 question_maxlength=40,
                 projection=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls=extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection

class Retriever(transformers.PreTrainedModel):

    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)
        assert config.projection or config.indexing_dimension == 768, \
            'If no projection then indexing dimension must be equal to 768'
        self.config = config
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        if self.config.projection:
            self.proj = nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )
            self.norm = nn.LayerNorm(self.config.indexing_dimension)
        self.loss_fct = torch.nn.KLDivLoss()

    def forward(self,
                question_ids,
                question_mask,
                passage_ids,
                passage_mask,
                gold_score=None):
        question_output = self.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_text(
            text_ids=passage_ids,
            text_mask=passage_mask,
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )

        score = torch.einsum(
            'bd,bid->bi',
            question_output,
            passage_output.view(bsz, n_passages, -1)
        )
        score = score / np.sqrt(question_output.size(-1))
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return question_output, passage_output, score, loss

    def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = self.model(
            input_ids=text_ids,
            attention_mask=text_mask if apply_mask else None
        )
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)
