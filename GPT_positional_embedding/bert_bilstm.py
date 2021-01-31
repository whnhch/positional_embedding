# -*- coding: utf-8 -*-
"""Positional_Encoding_At_Train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wDeaZFGaKu9i6Fy0rmJwvesdX3sg3huM

# **Implement Positional Encoding with GPT2 output**
----
"""

# #### **Install trsansformers** 

# package for using BERT from huggingface
# https://huggingface.co/transformers/index.html

# !pip install transformers

# for gpu
import torch
import math
import sys

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU avbailable, using the CPU instead')
    device = torch.device('cpu')

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

"""## **1. Implement GPT2 Decoder**"""

import transformers
import torch
from transformers import GPT2Model
from PIL import Image
import numpy as np

from transformers import BertTokenizer, GPT2Tokenizer

print('Loading BERT tokenizer...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

object_type_list = ['OBJORGANIZATION',
                    'OBJNATIONALITY',
                    'OBJMISC',
                    'OBJRELIGION',
                    'OBJNUMBER',
                    'OBJCRIMINALCHARGE',
                    'OBJIDEOLOGY',
                    'OBJSTATEORPROVINCE',
                    'OBJCITY',
                    'OBJCOUNTRY',
                    'OBJPERSON',
                    'OBJTITLE',
                    'OBJLOCATION',
                    'OBJCAUSEOFDEATH',
                    'OBJDURATION',
                    'OBJURL',
                    'OBJDATE']
subject_type_list = ['SUBJORGANIZATION', 'SUBJPERSON']

type_list = object_type_list + subject_type_list
bert_tokenizer.add_tokens(type_list, special_tokens=True)


def get_prob_embeds(probs, length, max_len, dim=768):
    embed = get_linear_embedding(max_len).to(device)

    for i in range(length - 1):
        for j in range(i + 1, length):
            width_ratio = 1 / (int(768 / 2))
            height_ratio = 2 / (max_len - 1)

            difference = (height_ratio - (height_ratio * width_ratio * j))
            cur_probs = probs[i][j]

            if cur_probs >= 0.0001:
                embed[j, :int(dim / 2)] = embed[j, :int(dim / 2)] + difference * cur_probs
                embed[j, int(dim / 2):] = embed[j, int(dim / 2):] - difference * cur_probs

            else:
                embed[j, :int(dim / 2)] = embed[j, :int(dim / 2)] - difference * (1 - cur_probs)
                embed[j, int(dim / 2):] = embed[j, int(dim / 2):] + difference * (1 - cur_probs)

    return embed


"""### **Positional Encoding with GPT2 output** """


def get_one_hot_encoding(max_len, dim=768):
    pe = torch.zeros(max_len, dim)

    for i in range(max_len):
        pe[i, i] = 1

    return pe


def get_linear_embedding(max_len, dim=768):
    isOdd = False

    if dim % 2 != 0:
        isOdd = True

    pe = torch.zeros(max_len, dim)

    width_ratio = 1 / (int(dim / 2))
    height_ratio = 2 / (max_len - 1)

    for i in range(dim):
        pe[0, i] = 1 - (2 / (dim - 1)) * i

    for i in range(1, max_len):
        for j in range(int(dim / 2)):
            pe[i, j] = pe[i - 1, j] - (height_ratio - (height_ratio * width_ratio * j))
            pe[i, dim - 1 - j] = pe[i - 1, dim - 1 - j] + (height_ratio - (height_ratio * width_ratio * j))
        if isOdd:
            pe[i, int(dim / 2) + 1] = pe[i - 1, int(dim / 2) + 1] - (height_ratio - (height_ratio * width_ratio * j))

    return pe


def get_sinusoidal_embedding(max_len, dim=768):
    pe = torch.zeros(max_len, dim)

    for pos in range(max_len):
        for i in range(0, dim, 2):
            pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i) / dim)))
            pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1)) / dim)))

    return pe


import pandas as pd
import numpy as np

label_list = ['org:member_of', 'per:schools_attended', 'per:charges', 'org:city_of_headquarters',
              'org:country_of_headquarters', 'org:subsidiaries', 'per:employee_of', 'per:stateorprovince_of_death',
              'per:stateorprovince_of_birth', 'per:country_of_death', 'org:shareholders', 'per:countries_of_residence',
              'per:children', 'org:alternate_names', 'per:alternate_names', 'per:stateorprovinces_of_residence',
              'per:country_of_birth', 'org:founded_by', 'org:parents', 'org:stateorprovince_of_headquarters',
              'org:dissolved', 'org:members', 'per:age', 'per:spouse', 'org:website', 'per:cities_of_residence',
              'per:parents', 'per:cause_of_death', 'per:date_of_death', 'per:origin', 'no_relation', 'per:religion',
              'org:political/religious_affiliation', 'per:siblings', 'org:founded', 'per:date_of_birth',
              'per:city_of_death', 'org:number_of_employees/members', 'org:top_members/employees', 'per:other_family',
              'per:title', 'per:city_of_birth'
              ]

train_df = pd.read_csv('./dataset/train_masked.tsv', delimiter='\t', header=None,
                       names=['sentence_id', 'label', 'label_notes', 'sentence', 'masked_sentence'])
dev_df = pd.read_csv('./dataset/dev_masked.tsv', delimiter='\t', header=None,
                     names=['sentence_id', 'label', 'label_notes', 'sentence', 'masked_sentence'])
test_df = pd.read_csv('./dataset/test_masked.tsv', delimiter='\t', header=None,
                      names=['sentence_id', 'label', 'label_notes', 'sentence', 'masked_sentence'])

"""### **Make torch dataloader**

Minimum of train sequence length should not be zero it means that there is any sequeence which size is over MAX_LEN
"""

print('open bert data...')

tr_input_ids = torch.load('./dataset/tr_input_ids.pt').to('cpu')
tr_attribute_masks = torch.load('./dataset/tr_attribute_masks.pt').to('cpu')
tr_positional_ids = torch.load('./dataset/tr_positional_ids.pt').to('cpu')
tr_seq_length = torch.load('./dataset/tr_seq_length.pt').to('cpu')
tr_gt_labels = torch.load('./dataset/tr_gt_labels.pt').to('cpu')

#
batch_size = 3
max_len = 450

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 3
max_len = 450
dim = 768

print('make train dataset...')

prob_ids = [x for x in range(len(tr_input_ids))]
train_dataset = TensorDataset(tr_input_ids, tr_attribute_masks, tr_positional_ids, tr_seq_length, tr_gt_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
#

# B batch size, deafult = 32
# P length of sequence, default = 450
# H hidden size of BERT, default = 768
# E embedding dimension of embedding layer, default = 20

from transformers import BertModel, AdamW, BertConfig
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BERT_BiLSTM(nn.Module):
    def __init__(self, tokenizer, max_length=450, num_labels=42, embedding_dim=20, num_recurrent_layers=1,
                 bidirectional=True, lstm_hidden_size=768, mlp_hidden_size=300) -> None:
        # super(BERT_BiLSTM, self).__init__()
        # Feed at [CLS] token

        super().__init__()
        self.tokenizer = tokenizer
        self.config = BertConfig.from_pretrained('bert-base-cased')
        self.config.output_hidden_states = True
        self.config.output_attentions = True

        self.bert = BertModel.from_pretrained("bert-base-cased",
                                              config=self.config
                                              # , add_pooling_layer=False
                                              )
        self.bert.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(num_embeddings=max_length * 2,
                                            embedding_dim=embedding_dim,
                                            padding_idx=0
                                            )

        self.num_recurrent_layers = num_recurrent_layers
        self.bidirectional = bidirectional
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(self.lstm_hidden_size + self.embedding_dim + self.embedding_dim,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=True,
                            bidirectional=True
                            )

        self.mlp = nn.Linear(in_features=self.lstm_hidden_size * 2, out_features=mlp_hidden_size)
        self.classifier = nn.Linear(in_features=mlp_hidden_size, out_features=self.num_labels)

    def forward(self, input_ids, attention_mask, positional_ids, seq_lengths, sequence_length, labels=None,
                position_embeds=None):
        # outputs: (last_encoder_layer, pooled_output, attention_weight)
        # Set add_pooling_layer
        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 encoder_hidden_states=torch.FloatTensor(
                                     (input_ids.shape[0], sequence_length, sequence_length)),
                                 position_embeds=position_embeds
                                 )

        sequence_output = bert_outputs[0]  # (B, P, H)

        for i in range(len(input_ids)):
            sequence_output[i, seq_lengths[i]:] = 0

        # LSTM
        ps = positional_ids[:, 0, :]
        po = positional_ids[:, 1, :]

        ps = self.embedding_layer(ps)
        po = self.embedding_layer(po)

        lstm_input = torch.cat((sequence_output, ps, po), dim=-1)  # (B, P, H+E+E)

        packed_lstm_input = pack_padded_sequence(lstm_input, seq_lengths.tolist(), batch_first=True,
                                                 enforce_sorted=False)

        h0 = torch.zeros(self.num_recurrent_layers * 2, input_ids.shape[0], self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.num_recurrent_layers * 2, input_ids.shape[0], self.lstm_hidden_size).to(device)

        packed_lstm_output, (lstm_hidden, lstm_cell) = self.lstm(packed_lstm_input, (h0, c0))  # (B, P, H*2) ...

        # mlp
        mlp_input = torch.cat((lstm_hidden[-2, :, :], lstm_hidden[-1, :, :]), dim=1)
        mlp_output = self.mlp(mlp_input)  # (B, H*2)

        # last layer
        # calcuate logits
        logits = self.classifier(mlp_output)

        # calcuate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss


model = BERT_BiLSTM(bert_tokenizer, max_length=max_len)
model.to(device)

"""### **Generate model**"""

batch_size = 3  # (set when creating our DataLoaders)
learning_rate = 1e-5
epochs = 3

optimizer = AdamW(model.parameters(),
                  lr=learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-12  # args.adam_epsilon  - default is 1e-8.
                  )

# cur_step = chkpt['step']
# cur_epoch = 0
# loss = chkpt['loss']

# optimizer.load_state_dict(chkpt['optimizer_state_dict'])

from transformers import get_linear_schedule_with_warmup

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)

import numpy as np
import torch.nn.functional as F


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


import random
import numpy as np
import torch.nn.functional as F
from datetime import timezone, timedelta
import sys
import os

seed_val = 42
dim = 768
max_len = 450

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

# decoder = GPTDecoder('gpt2-medium', max_len)
# decoder.to(device)
# decoder.eval()

max_len = 450

tz = timezone(timedelta(hours=9))
total_t0 = time.time()

MAX_LEN = 450

# For each epoch...
for epoch_i in range(epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()
    # Reset the total loss for this epoch.
    total_train_loss = []

    total_train_accuracy = []
    predictions_, true_labels_ = [], []

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_input_positional_ids = batch[2].to(device)
        b_input_seq_lengths = batch[3].to(device)
        b_labels = batch[4].to(device)

        model.zero_grad()

        cur_len = len(b_input_ids)

        b_input_seq_lengths, indicies = torch.sort(b_input_seq_lengths, dim=0, descending=True)


        def smart_sort(x, per):
            z = torch.empty_like(x)
            for i in range(len(per)):
                z[per[i]] = x[i]
            return z


        b_input_ids = smart_sort(b_input_ids, indicies)
        b_input_mask = smart_sort(b_input_mask, indicies)
        b_input_positional_ids = smart_sort(b_input_positional_ids, indicies)
        b_labels = smart_sort(b_labels, indicies)

        logits, loss = model(
            b_input_ids,
            attention_mask=b_input_mask,
            positional_ids=b_input_positional_ids,
            seq_lengths=b_input_seq_lengths,
            sequence_length=MAX_LEN,
            labels=b_labels,
            position_embeds=None
        )
        # (B, P)
        logits = F.softmax(logits, dim=1)

        total_train_loss.append(loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

        logits_ = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_train_accuracy.append(flat_accuracy(logits_, label_ids))

        # (B, P)

        if step % 16 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:},     Loss: {:}.      Accuracy: {:}'.format(step, len(
                train_dataloader), elapsed, sum(total_train_loss[:(step)]) / (step), sum(
                total_train_accuracy[:(step)]) / (step)))

    # Report the final accuracy for this validation run.
    avg_train_accuracy = sum(total_train_accuracy) / len(train_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_train_accuracy))

    # Calculate the average loss over all of the batches.
    avg_train_loss = sum(total_train_loss) / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    SAVE_EPOCH = epoch_i
    SAVE_PATH = "bert_bilstm_" + str(epoch_i) + ".pt"
    SAVE_LOSS = loss

    torch.save({
        'epoch': SAVE_EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': SAVE_LOSS,
        'step': step
    }, SAVE_PATH)