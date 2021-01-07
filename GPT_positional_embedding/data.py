import torch
import tokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd


def get_data(input_file, file_str):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        result = []
        masked_result = []
        for i in range(len(data)):
            id = data[i]['id']

            token = data[i]['token']
            masked_token = token[:]

            ss, se = data[i]['subj_start'], data[i]['subj_end']
            os, oe = data[i]['obj_start'], data[i]['obj_end']

            subject = ' '.join(token[ss:se + 1])
            object = ' '.join(token[os:oe + 1])

            masked_token[ss:se + 1] = ['SUBJ' + data[i]['subj_type'].replace('_', '')] * (se - ss + 1)
            masked_token[os:oe + 1] = ['OBJ' + data[i]['obj_type'].replace('_', '')] * (oe - os + 1)

            sentence = ' '.join(token)
            masked_sentence = ' '.join(masked_token)

            masked_sentence = '[CLS]' + masked_sentence + '[SEP]'
            masked_sentence += subject + '[SEP]' + object + '[SEP]'

            alpha = 'a'
            label = data[i]['relation']

            masked_result.append([id, label, alpha, sentence, masked_sentence])

        masked_df = pd.DataFrame(masked_result)

        masked_df.to_csv(file_str + '_masked.tsv', header=False, sep='\t', index=False)

        return masked_df


def get_data_loader(tokenizer, sentences, batch_size, labels, max_len, data_type='train', tokenizer_type='BERT'):
    dataset = None
    sampler = None

    if tokenizer_type.upper() == 'BERT':
        input_ids, attribute_masks, positional_ids, seq_length, gt_labels = tokenizer.get_bert_tokens(sentences, labels,
                                                                                                      tokenizer,
                                                                                                      max_len)
        dataset = TensorDataset(input_ids, attribute_masks, positional_ids, seq_length, gt_labels)

    else:
        input_ids, attribute_masks, positional_ids, seq_length, gt_labels = tokenizer.get_gpt_tokens(sentences, labels,
                                                                                                     tokenizer,
                                                                                                     max_len)
        dataset = TensorDataset(input_ids, attribute_masks, positional_ids, seq_length, gt_labels)

    if data_type.upper() == 'TRAIN':
        sampler = RandomSampler(dataset)

    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)
