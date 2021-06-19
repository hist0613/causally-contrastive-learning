import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

#Use triplet margin loss for CF robustness
class BertForCounterfactualRobustness(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        anchor_input_ids=None,
        anchor_attention_mask=None,
        positive_input_ids=None,
        positive_attention_mask=None,
        negative_input_ids=None,
        negative_attention_mask=None,
        triplet_sample_masks=None,
        lambda_weight=None,
        anchor_token_type_ids=None,
        positive_token_type_ids=None,
        negative_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False

        anchor_outputs = self.bert(
            anchor_input_ids,
            attention_mask=anchor_attention_mask,
            token_type_ids=anchor_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        #Sequence Classification
        pooled_output = anchor_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #Sequence Classification Loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #Triplet Margin Loss
        if positive_input_ids is not None and negative_input_ids is not None and labels is not None:
            positive_outputs = self.bert(
                positive_input_ids,
                attention_mask=positive_attention_mask,
                token_type_ids=positive_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            negative_outputs = self.bert(
                negative_input_ids,
                attention_mask=negative_attention_mask,
                token_type_ids=negative_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            ### TMP_MEMORY TEST ###
            positive_outputs_1 = self.bert(
                positive_input_ids,
                attention_mask=positive_attention_mask,
                token_type_ids=positive_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            negative_outputs_1 = self.bert(
                negative_input_ids,
                attention_mask=negative_attention_mask,
                token_type_ids=negative_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            #######################
            triplet_loss = None
            triplet_loss_fct = torch.nn.TripletMarginLoss()

            if lambda_weight is None:
                lambda_weight = 0.1

            if triplet_sample_masks is None:
                triplet_loss = triplet_loss_fct(anchor_outputs[1], positive_outputs[1], negative_outputs[1])
                loss = loss + lambda_weight * triplet_loss
            else:
                if torch.sum(triplet_sample_masks):
                    triplet_loss = triplet_loss_fct(anchor_outputs[1][triplet_sample_masks], positive_outputs[1][triplet_sample_masks], negative_outputs[1][triplet_sample_masks])
                    loss = loss + lambda_weight * triplet_loss

        if not return_dict:
            output = (logits,) + anchor_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        #return SequenceClassifierOutput(
        #    loss=loss,
        #    logits=logits,
        #    hidden_states=outputs.hidden_states,
        #    attentions=outputs.attentions,
        #)

"""
#Use triplet margin loss for CF robustness
class BertForCounterfactualRobustnessWithMasker(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.net_ssl = torch.nn.Sequential(  # self-supervision layer
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 30522),
        )


    def forward(
        self,
        anchor_input_ids=None,
        anchor_attention_mask=None,
        positive_input_ids=None,
        positive_attention_mask=None,
        negative_input_ids=None,
        negative_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        ssl_labels = None,
        uniform_labels = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_dict = False

        anchor_outputs = self.bert(
            anchor_input_ids,
            attention_mask=anchor_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        #Sequence Classification
        pooled_output = anchor_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #Sequence Classification Loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #Triplet Margin Loss
        if positive_input_ids is not None and negative_input_ids is not None and labels is not None:
            positive_outputs = self.bert(
                positive_input_ids,
                attention_mask=positive_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            negative_outputs = self.bert(
                negative_input_ids,
                attention_mask=negative_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            triplet_loss = None
            triplet_loss_fct = torch.nn.TripletMarginLoss()
            triplet_loss = triplet_loss_fct(anchor_outputs[1], positive_outputs[1], negative_outputs[1])
            #loss = loss + 0.1 * triplet_loss

            out_ssl = self.dropout(positive_outputs[0])
            out_ssl = self.net_ssl(out_ssl)
            out_ssl = out_ssl.permute(0, 2, 1)
            loss_ssl = F.cross_entropy(out_ssl, ssl_labels, ignore_index=-1)


            out_ood = self.dropout(negative_outputs[1])
            out_ood = self.classifier(out_ood)
            out_ood = F.log_softmax(out_ood, dim=1)  # log-probs
            loss_ent = F.kl_div(out_ood, uniform_labels)


            #loss = loss + 0.1 * triplet_loss + 0.001 * loss_ent
            loss = loss + 0.1 * triplet_loss + 0.001 * loss_ssl + 0.0001 * loss_ent
 


        if not return_dict:
            output = (logits,) + anchor_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        #return SequenceClassifierOutput(
        #    loss=loss,
        #    logits=logits,
        #    hidden_states=outputs.hidden_states,
        #    attentions=outputs.attentions,
        #)
"""
