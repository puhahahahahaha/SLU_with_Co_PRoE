import torch
import torch.nn as nn
from layers.masked_crf import CRF
from layers.utils import *
from transformers import BertPreTrainedModel, BertModel, BertConfig
from module import IntentClassifier, SlotClassifier, IntentTokenClassifier, MultiIntentClassifier, TagIntentClassifier
import logging

logger = logging.getLogger()


class JointBERTMultiIntent(BertPreTrainedModel):
    # multi_intent: 1,
    # intent_seq: 1,
    # tag_intent: 1,
    # bi_tag: 1,
    # cls_token_cat: 1,
    # intent_attn: 1,
    # num_mask: 4
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super().__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        # load pretrain bert
        # self.bert = BertModel(config=config)
        # self.label2id = Label2id(slot_label_lst)
        # self.bert = BertModel.from_pretrained("./bert-base-uncased", config=config)
        self.bert = BertModel.from_pretrained(args.model_name_or_path, config=config)

        # self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.multi_intent_classifier = MultiIntentClassifier(config.hidden_size, self.num_intent_labels,
                                                             args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)
        if args.intent_seq:
            self.intent_token_classifier = IntentTokenClassifier(config.hidden_size, self.num_intent_labels,
                                                                 args.dropout_rate)

        if args.tag_intent:
            if args.cls_token_cat:
                self.tag_intent_classifier = TagIntentClassifier(2 * config.hidden_size, self.num_intent_labels,
                                                                 args.dropout_rate)
            else:
                self.tag_intent_classifier = TagIntentClassifier(config.hidden_size, self.num_intent_labels,
                                                                 args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True, label2idx=self.label2id)

        self.intent_fc = nn.Linear(self.args.hidden_size, self.num_intent_labels)
        self.slot_fc = nn.Linear(self.args.hidden_size, self.num_slot_labels)

        self.I_S_Emb = Label_Attention(self.intent_fc, self.slot_fc)
        self.T_block1 = I_S_Block(self.intent_fc, self.slot_fc, self.args.hidden_size, self.args.dropout_rate,
                                  args.max_seq_len)
        self.T_block2 = I_S_Block(self.intent_fc, self.slot_fc, self.args.hidden_size, self.args.dropout_rate,
                                  args.max_seq_len)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                intent_label_ids,
                slot_labels_ids,
                # intent_token_ids,
                # B_tag_mask,
                # BI_tag_mask,
                # tag_intent_label
                ):
        """
            Args: B: batch_size; L: sequence length; I: the number of intents; M: number of mask; D: the output dim of Bert
            input_ids: B * L
            token_type_ids: B * L
            token_type_ids: B * L
            intent_label_ids: B * I
            slot_labels_ids: B * L
            intent_token_ids: B * L
            B_tag_mask: B * M * L
            BI_tag_mask: B * M * L
            tag_intent_label: B * M
        """
        # input_ids:  torch.Size([32, 50])
        # attention_mask:  torch.Size([32, 50])
        # token_type_ids:  torch.Size([32, 50])
        # intent_label_ids:  torch.Size([32, 10])
        # slot_labels_ids:  torch.Size([32, 50])
        # intent_token_ids:  torch.Size([32, 50])
        # B_tag_mask:  torch.Size([32, 4, 50])
        # BI_tag_mask:  torch.Size([32, 4, 50])
        # tag_intent_label:  torch.Size([32, 4])

        # (len_seq, batch_size, hidden_dim), (batch_size, hidden_dim)
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        # B * L * D
        sequence_output = outputs[0]

        # B * D
        H_I, H_S = self.I_S_Emb(sequence_output, sequence_output, attention_mask)
        H_I, H_S = self.T_block1(H_I + sequence_output, H_S + sequence_output, attention_mask)
        H_I_1, H_S_1 = self.I_S_Emb(H_I, H_S, attention_mask)
        sequence_output_I, sequence_output_S = self.T_block2(H_I + H_I_1, H_S + H_S_1, attention_mask)

        # H_I = sequence_output
        # H_S = sequence_output
        # H_I, H_S = self.T_block1(H_I, H_S, attention_mask)
        # sequence_output_I, sequence_output_S = self.T_block2(H_I + sequence_output, H_S + sequence_output, attention_mask)

        pooled_output = torch.nn.functional.max_pool1d(sequence_output_I.transpose(1, 2),
                                                       sequence_output_I.size(1)).squeeze(2)

        total_loss = 0

        intent_loss = 0.0
        slot_loss = 0.0

        # ==================================== 1. Intent Softmax ========================================
        # (batch_size, num_intents)
        intent_logits = self.multi_intent_classifier(pooled_output)
        intent_logits_cpu = intent_logits.data.cpu().numpy()

        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1, self.num_intent_labels))
            else:
                # intent_loss_fct = nn.CrossEntropyLoss()
                # default reduction is mean
                intent_loss_fct = nn.BCELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels) + 1e-10,
                                              intent_label_ids.view(-1, self.num_intent_labels))
            # Question: do we need to add weight here
            total_loss += intent_loss

        # if intent_label_ids.type() != torch.cuda.FloatTensor:
        #     intent_label_ids = intent_label_ids.type(torch.cuda.FloatTensor)

        # ==================================== 2. Slot Softmax ========================================
        # (batch_size, seq_len, num_slots)
        slot_logits = self.slot_classifier(sequence_output_S)

        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    try:
                        active_loss = attention_mask.view(-1) == 1
                        attention_mask_cpu = attention_mask.data.cpu().numpy()
                        active_loss_cpu = active_loss.data.cpu().numpy()
                        active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                        active_labels = slot_labels_ids.view(-1)[active_loss]
                        slot_loss = slot_loss_fct(active_logits, active_labels)
                    except:
                        print('intent_logits: ', intent_logits_cpu)
                        print('attention_mask: ', attention_mask_cpu)
                        print('active_loss: ', active_loss_cpu)
                        logger.info('intent_logits: ', intent_logits_cpu)
                        logger.info('attention_mask: ', attention_mask_cpu)
                        logger.info('active_loss: ', active_loss_cpu)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]

        outputs = ([total_loss, intent_loss, slot_loss],) + outputs


        return outputs  # (loss), logits, (hidden_states), (attentions), [attention_probs_intent, attention_probs_slot] # Logits is a tuple of intent and slot logits
