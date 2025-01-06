# coding=utf-8
# copied from bart

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel
from transformers.generation_utils import top_k_top_p_filtering
from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration,BlenderbotSmallModel )
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput, )
from .PARAMS import SAMPLE, TEMPERATURE
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import math


def to_var(x, on_cpu=False, gpu_id=None, async=False):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id, async)
        # x = Variable(x)
    return x


def normal_logpdf(x, mean, var):
    """
    Args:
        x: (Variable, FloatTensor) [batch_size, dim]
        mean: (Variable, FloatTensor) [batch_size, dim] or [batch_size] or [1]
        var: (Variable, FloatTensor) [batch_size, dim]: positive value
    Return:
        log_p: (Variable, FloatTensor) [batch_size]
    """

    pi = to_var(torch.FloatTensor([np.pi]))
    return 0.5 * torch.sum(-torch.log(2.0 * pi) - torch.log(var) - ((x - mean).pow(2) / var), dim=1)


def normal_kl_div(mu1, var1,
                  mu2=to_var(torch.FloatTensor([0.0])),
                  var2=to_var(torch.FloatTensor([1.0]))):
    one = to_var(torch.FloatTensor([1.0]))
    return torch.sum(0.5 * (torch.log(var2) - torch.log(var1)
                            + (var1 + (mu1 - mu2).pow(2)) / var2 - one), 1)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class FeedForward(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, hidden_size=None,
                 activation="Tanh", bias=True):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = getattr(nn, activation)()
        n_inputs = [input_size] + [hidden_size] * (num_layers - 1)
        n_outputs = [hidden_size] * (num_layers - 1) + [output_size]
        self.linears = nn.ModuleList([nn.Linear(n_in, n_out, bias=bias)
                                      for n_in, n_out in zip(n_inputs, n_outputs)])

    def forward(self, input):
        x = input
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)

        return x


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self.emotion_embedding = nn.Embedding(num_embeddings=28, embedding_dim=self.model.config.d_model)
        self.context_norm = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        self.comet_norm = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        self.strat_norm = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        self.dropout = nn.Dropout(0.65)
        self.fc1 = nn.Linear(self.model.config.d_model, self.model.config.d_model)
        self.fc2 = nn.Linear(self.model.config.d_model, self.model.config.d_model)
        # self.linear = nn.Linear(self.model.config.d_model*2, self.model.config.d_model)
        self.relu=nn.ReLU()
        self.gru=nn.GRU(self.model.config.d_model, self.model.config.d_model,1,batch_first=True)
        # self.persona_comet_w = nn.Parameter(torch.tensor([1 / 2, 1 / 2]))
        self.persona_context_w = nn.Parameter(torch.tensor([1 / 3, 1 / 3,1/3]))
        # self.w1 = nn.Parameter(torch.tensor(1 / 2), requires_grad=True)
        # self.w2 = nn.Parameter(torch.tensor(1 / 2), requires_grad=True)
        self.softplus = nn.Softplus()
        self.persona_infer =True
        # self.sys_state=False
        self.usr_state = True
        self.persona_input =True
        self.persona_cat=True
        self.reframing =True
        self.strategy_pre = True
        # self.vae_decoder=BlenderbotSmallModel(self.model.config).decoder
        # self.fc = PositionalWiseFeedForward(self.hidden_size, self.embed_dim, self.dropout_rate, 'gelu')
        # self.cnn = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3)
        # self.max_pool = nn.MaxPool1d(kernel_size=3)
        # self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.prior_h_usr = FeedForward(self.model.config.d_model,
                                       self.model.config.d_model,
                                       num_layers=1,
                                       hidden_size=self.model.config.d_model,
                                       )
        self.prior_mu_usr = nn.Linear(self.model.config.d_model,
                                      self.model.config.d_model)
        self.prior_var_usr = nn.Linear(self.model.config.d_model,
                                       self.model.config.d_model)
        self.posterior_h = FeedForward(self.model.config.d_model,
                                   self.model.config.d_model,
                                   num_layers=1,
                                   hidden_size=self.model.config.d_model,
                                   )
        self.posterior_mu = nn.Linear(self.model.config.d_model,
                                  self.model.config.d_model)
        self.posterior_var = nn.Linear(self.model.config.d_model,
                                   self.model.config.d_model)

    def prior_usr(self, context_outputs):
        # Context dependent prior
        h_prior = self.prior_h_usr(context_outputs)
        mu_prior = self.prior_mu_usr(h_prior)
        var_prior = self.softplus(self.prior_var_usr(h_prior))
        return mu_prior, var_prior

    # def prior(self, context_outputs):
    #     # Context dependent prior
    #     h_prior = self.prior_h(context_outputs)
    #     mu_prior = self.prior_mu(h_prior)
    #     var_prior = self.softplus(self.prior_var(h_prior))
    #     return mu_prior, var_prior
    #
    def posterior(self, encoder_hidden):
        # Context dependent prior
        h_posterior =self.posterior_h(encoder_hidden)
        mu_posterior = self.posterior_mu(h_posterior)
        var_posterior = self.softplus(self.posterior_var(h_posterior))
        return mu_posterior, var_posterior
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_input_embeds=None,
            encoder_outputs=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            return_dict=None,
            validation=False,
            persona_ids=None,
            persona_attention_mask=None,
            final_persona_ids=None,
            final_persona_attention_mask=None,
            comet_ids=None,
            comet_attention_mask=None,
            reframing_ids=None,
            reframing_attention_mask=None,
            decoder_attention_mask=None,
            str_bos=None,
            strat_id=None,
            sys_begin_ids=None,
            sys_input_turns=None,
            usr_begin_ids=None,
            usr_input_turns=None,
            supportor_id=None,
            seeker_id=None,
            last_input_ids=None,
            last_labels=None,
            last_attention_mask=None,
            pad_token_id=-100,
            emotion=None,
            emotion_score=None,
            **kwargs
    ):
        # persona_ids = final_persona_ids
        # persona_attention_mask =final_persona_attention_mask
        bsz=decoder_input_ids.size()[0]
        assert self.toker is not None

        encoded_info = kwargs
        assert (self.training or validation) == (labels is not None)
        if validation:
            labels[:, 0] = -100

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:  # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = self.model.config.output_attentions
        output_hidden_states = (
            self.model.config.output_hidden_states
        )
        if encoder_outputs is not None:
            # print(input_ids, attention_mask, decoder_input_ids, encoder_outputs, (past_key_values is None))
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=return_dict,
            )
        else:
            head_mask = None
            inputs_embeds = None
            cross_attn_head_mask = None
            decoder_head_mask = None
            decoder_attention_mask = None
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            original_encoder_states = encoder_outputs.last_hidden_state
            if persona_ids is not None:
                persona_embeds=self.model.shared(persona_ids)
                persona_encoder_outputs = self.model.encoder(
                    input_ids=None,
                    attention_mask=persona_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=persona_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            if self.persona_infer == True:
                comet_encoder_outputs = self.model.encoder(
                    input_ids=comet_ids,
                    attention_mask=comet_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                comet_encoder_outputs = None
            # plan1
            if self.persona_cat == True:
                encoder_outputs.last_hidden_state = torch.cat(
                    [encoder_outputs.last_hidden_state, persona_encoder_outputs.last_hidden_state], dim=1)
                attention_mask = torch.cat([attention_mask, persona_attention_mask], dim=1)
            if self.persona_infer == True:
                encoder_outputs.last_hidden_state = torch.cat([encoder_outputs.last_hidden_state, comet_encoder_outputs.last_hidden_state], dim=1)
                attention_mask = torch.cat([attention_mask, comet_attention_mask], dim=1)
            # plan2
            # w1 = torch.exp(self.persona_context_w[0]) / torch.sum(torch.exp(self.persona_context_w))
            # w2 = torch.exp(self.persona_context_w[1]) / torch.sum(torch.exp(self.persona_context_w))
            # # w3 = torch.exp(self.persona_comet_w[0]) / torch.sum(torch.exp(self.persona_comet_w))
            # # w4 = torch.exp(self.persona_comet_w[1]) / torch.sum(torch.exp(self.persona_comet_w))
            # if self.persona_input == True:
            #         # context = torch.stack([torch.matmul(torch.softmax(torch.matmul(j, i.t()), dim=-1), i) for i, j in
            #         #                    zip(encoder_outputs[0], persona_encoder_outputs[0])])
            #     persona_attention_mask_ = _expand_mask(persona_attention_mask, encoder_outputs[0].dtype,
            #                                                tgt_len=input_ids.size()[-1]).squeeze(1)
            #     persona1 = torch.stack(
            #             [torch.matmul(torch.softmax(torch.matmul(i, j.t()) + m, dim=-1), j) for i, j, m in
            #              zip(encoder_outputs[0], persona_encoder_outputs[0], persona_attention_mask_)])
            #     persona1 = self.context_norm(encoder_outputs.last_hidden_state + persona1)
            #     encoder_outputs1 = w1 * encoder_outputs.last_hidden_state + w2 * persona1
            # if self.persona_infer == True:
            #     comet_attention_mask_ = _expand_mask(persona_attention_mask, encoder_outputs[0].dtype,
            #                                              tgt_len=comet_ids.size()[-1]).squeeze(1)
            #     persona2 = torch.stack(
            #             [torch.matmul(torch.softmax(torch.matmul(i, j.t()) + m, dim=-1), j) for i, j, m in
            #              zip(comet_encoder_outputs[0], persona_encoder_outputs[0],
            #                  comet_attention_mask_)])  # ,persona_encoder_outputs[0]
            #     persona2 = self.comet_norm(comet_encoder_outputs.last_hidden_state + persona2)
            #     encoder_outputs2 = w1 * comet_encoder_outputs.last_hidden_state + w2 * persona2
                # 归一化权重
            # if self.persona_infer == True and self.persona_input == True:
            #         encoder_outputs.last_hidden_state = torch.cat([encoder_outputs1, encoder_outputs2], dim=1)
            #         attention_mask = torch.cat([attention_mask, comet_attention_mask], dim=1)
            # elif self.persona_infer == False and self.persona_input == True:
            #         encoder_outputs.last_hidden_state = encoder_outputs1
            # elif self.persona_infer == True and self.persona_input == False:
            #         encoder_outputs.last_hidden_state = torch.cat(
            #             [encoder_outputs.last_hidden_state, comet_encoder_outputs.last_hidden_state], dim=1)
            #         attention_mask = torch.cat([attention_mask, comet_attention_mask], dim=1)
            # plan3
            # w1 = torch.exp(self.persona_context_w[0]) / torch.sum(torch.exp(self.persona_context_w))
            # w2 = torch.exp(self.persona_context_w[1]) / torch.sum(torch.exp(self.persona_context_w))
            # w3 = torch.exp(self.persona_context_w[2]) / torch.sum(torch.exp(self.persona_context_w))
            # # w4 = torch.exp(self.persona_comet_w[1]) / torch.sum(torch.exp(self.persona_comet_w))
            # encoder_outputs_new=encoder_outputs.last_hidden_state
            # if self.persona_input == True:
            #     # context = torch.stack([torch.matmul(torch.softmax(torch.matmul(j, i.t()), dim=-1), i) for i, j in
            #     #                    zip(encoder_outputs[0], persona_encoder_outputs[0])])
            #     persona_attention_mask_ = _expand_mask(persona_attention_mask, encoder_outputs[0].dtype,
            #                                            tgt_len=input_ids.size()[-1]).squeeze(1)
            #     persona1 = torch.stack(
            #         [torch.matmul(torch.softmax(torch.matmul(i, j.t()) + m, dim=-1), j) for i, j, m in
            #          zip(original_encoder_states, persona_encoder_outputs[0], persona_attention_mask_)])
            #     persona1 = self.context_norm(encoder_outputs.last_hidden_state + persona1)
            #     encoder_outputs_new = w1 *encoder_outputs_new + w2 * persona1
            # if self.persona_infer == True:
            #     comet_attention_mask_ = _expand_mask(comet_attention_mask, encoder_outputs[0].dtype,
            #                                          tgt_len=input_ids.size()[-1]).squeeze(1)
            #     persona2 = torch.stack(
            #         [torch.matmul(torch.softmax(torch.matmul(i, j.t()) + m, dim=-1), j) for i, j, m in
            #          zip(original_encoder_states,comet_encoder_outputs[0],
            #              comet_attention_mask_)])  # ,persona_encoder_outputs[0]
            #     persona2 = self.comet_norm(encoder_outputs.last_hidden_state + persona2)
            #     encoder_outputs_new = encoder_outputs_new+ w3 * persona2
            # encoder_outputs.last_hidden_state=encoder_outputs_new

            # 归一化权重
            if self.reframing == True:
                reframing_encoder_outputs = self.model.encoder(
                    input_ids=reframing_ids,
                    attention_mask=reframing_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                reframing_hidden_states = reframing_encoder_outputs.last_hidden_state
                reframing_hidden_states = torch.mul(reframing_hidden_states, reframing_attention_mask.unsqueeze(2))
                reframing_states = torch.mean(reframing_hidden_states, dim=-2)
            if self.usr_state == True:
                # seeker_states= torch.stack([pad(context.narrow(0, s, l), max_len)  # narrow：取某一维度的几个值start开始长度为length
                #                        for s, l in zip(start.data.tolist(),
                #                                        input_conversation_length.data.tolist())], 0)
                sem_cls_embs = []
                for item, idx in zip(original_encoder_states, usr_begin_ids):  # trans_cls_index
                    cls_emb = torch.index_select(item, 0, idx)
                    sem_cls_embs.append(cls_emb)
                # for item in sem_cls_embs:
                sem_cls_embs = pad_sequence(sem_cls_embs, batch_first=True, padding_value=0)

                # seeker_states,_=self.gru(sem_cls_embs,seeker_init_hidden.unsqueeze(0))
                batch_size = sem_cls_embs.size(0)
                # persona_feature = persona_encoder_outputs.last_hidden_state[:, 0, :]
                persona_feature = torch.cat([sem_cls_embs[i, l - 1, :].unsqueeze(0)
                                             for i, l in enumerate(usr_input_turns.data)])
                mu_prior, var_prior = self.prior_usr(persona_feature)
                eps = torch.randn((int(batch_size), self.model.config.d_model)).to(persona_feature.device)
                persona_feature_usr = mu_prior + torch.sqrt(var_prior) * eps
                persona_feature_usr = persona_feature_usr.view(bsz, 1, -1)
                persona_feature_usr_msk = torch.ones(bsz, 1).to("cuda")
                # input_length = input_ids.size()[1]
                # encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state + \
                #                                     persona_feature_usr.repeat(1,input_length,1)
                if self.training:
                    last_input_embeds=self.model.shared(last_input_ids)
                    last_input_embeds[:,0]=persona_feature_usr.squeeze(1)
                    vae_outputs = self.model.decoder(
                        input_ids=None,
                        inputs_embeds=last_input_embeds,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        return_dict=return_dict,
                    )
                    # posterior_persona_feature=torch.mean(vae_outputs.last_hidden_state,dim=1)
                    # posterior_persona_feature=
                    # mu_posterior, var_posterior = self.posterior(posterior_persona_feature)
                    lm_logits_vae = self.lm_head(vae_outputs.last_hidden_state)
                    lm_logits_vae_flat_shifted = lm_logits_vae.contiguous().view(-1, lm_logits_vae.size(-1))
                    lm_labels_vae_flat_shifted = last_labels.contiguous().view(-1)
                    vae_loss = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)(lm_logits_vae_flat_shifted,
                                                                                    lm_labels_vae_flat_shifted)
                    seeker_ids=seeker_id.repeat(bsz).to("cuda")
                    seeker_init_hidden=self.model.shared(seeker_ids).unsqueeze(1)
                    # seeker_init_hidden=self.fc2(seeker_init_hidden)
                    seeker_attention_mask = _expand_mask(last_attention_mask, vae_outputs.last_hidden_state.dtype,
                                                            tgt_len=seeker_init_hidden.size()[1]).squeeze(1)
                    posterior_persona_feature = torch.stack(
                        [torch.matmul(torch.softmax(torch.matmul(i, j.t()) + m, dim=-1), j) for i, j, m in
                         zip(seeker_init_hidden, vae_outputs[0],
                             seeker_attention_mask)])
                    mu_posterior, var_posterior = self.posterior(posterior_persona_feature)
                    kl_div=normal_kl_div(mu_prior, var_prior,mu_posterior, var_posterior)
                    kl_div = torch.mean(kl_div)
                    # kl_div = torch.zeros(1).to('cuda')[0]
            else:
                vae_loss = torch.zeros(1).to('cuda')[0]
                kl_div = torch.zeros(1).to('cuda')[0]
            decoder_inputs_embeds = self.model.shared(decoder_input_ids)
            if self.usr_state == True:
                decoder_inputs_embeds = torch.cat([persona_feature_usr, decoder_inputs_embeds], dim=1)
            if self.reframing == True:
                # decoder_inputs_embeds[:, 0] = reframing_states  # .squeeze(1)
                decoder_inputs_embeds = torch.cat([reframing_states.unsqueeze(1), decoder_inputs_embeds], dim=1)
            outputs = self.model.decoder(
                input_ids=None,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if self.usr_state == True:
                outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
            if self.reframing == True:
                outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
            if self.strategy_pre == True:
                strategy_outputs = outputs.last_hidden_state[:, 0, :].unsqueeze(1)
                outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
                if self.persona_input==True:
                    persona_attention_mask_3 = _expand_mask(persona_attention_mask, encoder_outputs[0].dtype,
                                                            tgt_len=strategy_outputs.size()[1]).squeeze(1)
                    persona3 = torch.stack(
                        [torch.matmul(torch.softmax(torch.matmul(i, j.t()) + m, dim=-1), j) for i, j, m in
                         zip(strategy_outputs, persona_encoder_outputs[0],
                             persona_attention_mask_3)])
                    strat_states = self.fc1(strategy_outputs + persona3)
                    strat_states = self.strat_norm(strat_states)
                else:
                    strat_states=strategy_outputs
                # strat_states=self.fc1(strategy_outputs)
                # strat_states=self.dropout(self.relu(strat_states))
                # strat_states = self.fc2(strat_states)
                # persona_hidden_states = torch.mul(persona_encoder_outputs[0],
                #                                   persona_attention_mask.unsqueeze(2))
                # persona3 = torch.mean(persona_hidden_states, dim=-2)
                # strat_states = self.fc1(strategy_outputs + persona3.unsqueeze(1))
                # strat_states = self.strat_norm(strat_states)
                # emotion_embedding = self.emotion_embedding(emotion)
                # emotion_score = emotion_score.unsqueeze(1).repeat(1, self.model.config.d_model)
                # current_emotion = torch.mul(emotion_score, emotion_embedding)
                # strat_states=self.linear(current_emotion.unsqueeze(1))
                # strat_states = self.strat_norm(strategy_outputs + self.dropout(strat_states))
                strategy_logits = self.lm_head(strat_states)  # + self.final_logits_bias
                strategy_loss = nn.CrossEntropyLoss()(
                    strategy_logits.contiguous().view(-1, strategy_logits.size(-1)), strat_id.view(-1))
            else:
                strategy_loss = torch.zeros(1).to('cuda')[0]
        lm_logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias
        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()
        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if not self.training and not validation:  # inference
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        elif self.training:  # training
            assert not validation

            res = {'lm_loss': masked_lm_loss, 'ppl': ppl_value, 'strategy_loss': strategy_loss, 'vae_loss': vae_loss,
                   'kl_div': kl_div}
            return res

        else:  # validation
            assert not self.training
            return loss, label_size

    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        logits = logits[:, 0, -8:]

        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)

        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]

        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1)
        })

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids=None,
        decoder_input_embeds=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None, # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "decoder_input_embeds": decoder_input_embeds,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
    @torch.no_grad()
    def generate(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            comet_ids=None,
            comet_attention_mask=None,
            persona_ids=None,
            persona_attention_mask=None,
            final_persona_ids=None,
            final_persona_attention_mask=None,
            return_dict=None,
            reframing_ids=None,
            reframing_attention_mask=None,
            str_bos=None,
            usr_begin_ids=None,
            usr_input_turns=None,
            sys_begin_ids=None,
            sys_input_turns=None,
            seeker_id=None,
            emotion=None,
            emotion_score=None,
            **kwargs
    ):
        # persona_ids = final_persona_ids
        # persona_attention_mask = final_persona_attention_mask

        assert not self.training
        assert self.toker is not None

        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        original_encoder_states = encoder_outputs.last_hidden_state
        if persona_ids is not None:
            persona_embeds = self.model.shared(persona_ids)
            persona_encoder_outputs = self.model.encoder(
                input_ids=None,
                attention_mask=persona_attention_mask,
                inputs_embeds=persona_embeds,
                return_dict=return_dict,
            )
        if self.persona_infer == True:
            comet_encoder_outputs = self.model.encoder(
                input_ids=comet_ids,
                attention_mask=comet_attention_mask,
                return_dict=return_dict,
            )
        else:
            comet_encoder_outputs = None
        # plan1
        if self.persona_cat == True:
            encoder_outputs.last_hidden_state = torch.cat(
                [encoder_outputs.last_hidden_state, persona_encoder_outputs.last_hidden_state], dim=1)
            attention_mask = torch.cat([attention_mask, persona_attention_mask], dim=1)
        if self.persona_infer == True:
            encoder_outputs.last_hidden_state = torch.cat(
                [encoder_outputs.last_hidden_state, comet_encoder_outputs.last_hidden_state], dim=1)
            attention_mask = torch.cat([attention_mask, comet_attention_mask], dim=1)
        # plan2
        # w1 = torch.exp(self.persona_context_w[0]) / torch.sum(torch.exp(self.persona_context_w))
        # w2 = torch.exp(self.persona_context_w[1]) / torch.sum(torch.exp(self.persona_context_w))
        # # w3 = torch.exp(self.persona_comet_w[0]) / torch.sum(torch.exp(self.persona_comet_w))
        # # w4 = torch.exp(self.persona_comet_w[1]) / torch.sum(torch.exp(self.persona_comet_w))
        # if self.persona_input == True:
        #     # context = torch.stack([torch.matmul(torch.softmax(torch.matmul(j, i.t()), dim=-1), i) for i, j in
        #     #                    zip(encoder_outputs[0], persona_encoder_outputs[0])])
        #     persona_attention_mask_ = _expand_mask(persona_attention_mask, encoder_outputs[0].dtype,
        #                                            tgt_len=input_ids.size()[-1]).squeeze(1)
        #     persona1 = torch.stack(
        #         [torch.matmul(torch.softmax(torch.matmul(i, j.t()) + m, dim=-1), j) for i, j, m in
        #          zip(encoder_outputs[0], persona_encoder_outputs[0], persona_attention_mask_)])
        #     persona1 = self.context_norm(encoder_outputs.last_hidden_state + persona1)
        #     encoder_outputs1 = w1 * encoder_outputs.last_hidden_state + w2 * persona1
        # if self.persona_infer == True:
        #     comet_attention_mask_ = _expand_mask(persona_attention_mask, encoder_outputs[0].dtype,
        #                                          tgt_len=comet_ids.size()[-1]).squeeze(1)
        #     persona2 = torch.stack(
        #         [torch.matmul(torch.softmax(torch.matmul(i, j.t()) + m, dim=-1), j) for i, j, m in
        #          zip(comet_encoder_outputs[0], persona_encoder_outputs[0],
        #              comet_attention_mask_)])  # ,persona_encoder_outputs[0]
        #     persona2 = self.comet_norm(comet_encoder_outputs.last_hidden_state + persona2)
        #     encoder_outputs2 = w1 * comet_encoder_outputs.last_hidden_state + w2 * persona2
        #     # 归一化权重
        # if self.persona_infer == True and self.persona_input == True:
        #     encoder_outputs.last_hidden_state = torch.cat([encoder_outputs1, encoder_outputs2], dim=1)
        #     attention_mask = torch.cat([attention_mask, comet_attention_mask], dim=1)
        # elif self.persona_infer == False and self.persona_input == True:
        #     encoder_outputs.last_hidden_state = encoder_outputs1
        # elif self.persona_infer == True and self.persona_input == False:
        #     encoder_outputs.last_hidden_state = torch.cat(
        #         [encoder_outputs.last_hidden_state, comet_encoder_outputs.last_hidden_state], dim=1)
        #     attention_mask = torch.cat([attention_mask, comet_attention_mask], dim=1)
        # if self.sys_state==True:
        #     sem_cls_embs = []
        #     for item, idx in zip(original_encoder_states, sys_begin_ids):  # trans_cls_index
        #         cls_emb = torch.index_select(item, 0, idx)
        #         sem_cls_embs.append(cls_emb)
        #     sem_cls_embs = pad_sequence(sem_cls_embs, batch_first=True, padding_value=0)
        #     batch_size = sem_cls_embs.size(0)
        #     persona_feature = torch.cat([sem_cls_embs[i, l - 1, :].unsqueeze(0)
        #                                  for i, l in enumerate(sys_input_turns.data)])
        #     mu_prior, var_prior = self.prior(persona_feature)
        #     eps = torch.randn((int(batch_size), self.model.config.d_model)).to(persona_feature.device)
        #     persona_feature_z = mu_prior + torch.sqrt(var_prior) * eps
        if self.usr_state == True:
            sem_cls_embs = []
            for item, idx in zip(original_encoder_states, usr_begin_ids):  # trans_cls_index
                cls_emb = torch.index_select(item, 0, idx)
                sem_cls_embs.append(cls_emb)
            sem_cls_embs = pad_sequence(sem_cls_embs, batch_first=True, padding_value=0)
            batch_size = sem_cls_embs.size(0)
            # seeker_ids = seeker_id.repeat(batch_size).to("cuda")
            # seeker_init_hidden = self.model.shared(seeker_ids)
            # seeker_states, _ = self.gru(sem_cls_embs, seeker_init_hidden.unsqueeze(0))

            persona_feature = torch.cat([sem_cls_embs[i, l - 1, :].unsqueeze(0)
                                         for i, l in enumerate(usr_input_turns.data)])
            mu_prior, var_prior = self.prior_usr(persona_feature)
            # mu_posterior, var_posterior = self.posterior(persona_feature,
            #                                              response_encoder_outputs.last_hidden_state[:, 0])
            eps = torch.randn((int(batch_size), self.model.config.d_model)).to(persona_feature.device)
            persona_feature_usr = mu_prior + torch.sqrt(var_prior) * eps
            # persona_feature_usr = persona_feature_usr.view(bsz, 1, -1)
            # input_length=input_ids.size()[1]
            # encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state + persona_feature_usr.repeat(1,input_length,1)
        if self.reframing == True:
            reframing_encoder_outputs = self.model.encoder(
                input_ids=reframing_ids,
                attention_mask=reframing_attention_mask,
                inputs_embeds=None,
                return_dict=return_dict,
            )
            reframing_hidden_states = reframing_encoder_outputs.last_hidden_state
            # reframing_states=reframing_hidden_states[:,0]
            reframing_hidden_states = torch.mul(reframing_hidden_states, reframing_attention_mask.unsqueeze(2))
            reframing_states = torch.mean(reframing_hidden_states, dim=-2)
        decoder_inputs_embeds = self.model.decoder.embed_tokens(decoder_input_ids)
        if self.usr_state == True:
            decoder_inputs_embeds = torch.cat([persona_feature_usr.unsqueeze(1),decoder_inputs_embeds], dim=1)
        if self.reframing == True:
            # decoder_inputs_embeds[:, 0] = reframing_states  # .squeeze(1)
            decoder_inputs_embeds = torch.cat([reframing_states.unsqueeze(1), decoder_inputs_embeds], dim=1)

        strategy_outputs = self.model.decoder(
            input_ids=None,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        )
        past_key_values = strategy_outputs.past_key_values
        strategy_outputs = strategy_outputs.last_hidden_state[:, -1, :].unsqueeze(1)
        if self.strategy_pre==True:
            if self.persona_input == True:
                persona_attention_mask_3 = _expand_mask(attention_mask, encoder_outputs[0].dtype,
                                                        tgt_len=strategy_outputs.size()[1]).squeeze(1)
                persona3 = torch.stack([torch.matmul(torch.softmax(torch.matmul(i, j.t()) + m, dim=-1), j) for i, j, m in
                                        zip(strategy_outputs, encoder_outputs[0], persona_attention_mask_3)])
                strat_states = self.fc1(strategy_outputs + persona3)
                strat_states = self.strat_norm(strat_states)
            # persona_hidden_states = torch.mul(persona_encoder_outputs[0],
            #                                   persona_attention_mask.unsqueeze(2))
            # persona3 = torch.mean(persona_hidden_states, dim=-2)
            # strat_states = self.fc1(strategy_outputs + persona3.unsqueeze(1))
            # strat_states = self.strat_norm(strat_states)
            # emotion_embedding = self.emotion_embedding(emotion)
            # emotion_score = emotion_score.unsqueeze(1).repeat(1, self.model.config.d_model)
            # current_emotion = torch.mul(emotion_score, emotion_embedding)
            # strat_states = self.linear(current_emotion.unsqueeze(1))
            # strat_states = self.strat_norm(strategy_outputs + self.dropout(strat_states))
            # strategy_logits = self.lm_head(strat_states)  # + self.final_logits_bias
            # strat_states = self.fc1(strategy_outputs)
            # strat_states = self.dropout(self.relu(strat_states))
            # strat_states=self.fc2(strat_states)
            # strat_states = self.strat_norm(strategy_outputs + self.dropout(strat_states))
            else:
                strat_states=strategy_outputs
        strategy_logits = self.lm_head(strat_states)
        self.predict_strategy(strategy_logits, encoded_info)
        strategy_outputs_id = torch.argmax(strategy_logits, dim=-1)
        strategy_outputs_embed = self.model.decoder.embed_tokens(strategy_outputs_id)
        decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None]], dim=-1)
        decoder_input_embeds = torch.cat([decoder_inputs_embeds, strategy_outputs_embed], dim=1)

        # for i in range(kwargs["max_length"] - 1):
        #     decoder_outputs = self.model.decoder(
        #         input_ids=None,
        #         inputs_embeds=decoder_inputs_embeds,
        #         encoder_hidden_states=encoder_outputs[0],
        #         encoder_attention_mask=attention_mask,
        #         return_dict=return_dict,
        #     )
        #     # lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        #     lm_logits = self.lm_head(decoder_outputs.last_hidden_state)[:, -1, :]  # + self.final_logits_bias
        #     decoder_outputs_id = torch.argmax(lm_logits[:, :self.toker.vocab_size], dim=-1)
        #     decoder_outputs_embed = self.model.decoder.embed_tokens(decoder_outputs_id)
        #     decoder_input_ids = torch.cat([decoder_input_ids, decoder_outputs_id.unsqueeze(1)], dim=-1)
        #     decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, decoder_outputs_embed.unsqueeze(1)], dim=1)

        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True

        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids

        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            past=past_key_values,
            **kwargs
        )
        encoded_info['persona'] = persona_ids
        return encoded_info,generations[:, decoder_input_ids.size(1):]