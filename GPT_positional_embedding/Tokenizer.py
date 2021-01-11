import torch
from transformers import GPTTokenizer, BERTTokenizer

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


def get_gpt_tokenizer(gpt_model_name):
    return GPTTokenizer.from_pretrained(gpt_model_name)


def get_bert_tokenizer():
    return BERTTokenizer.from_pretrained('bert-base-cased')


# gpt tokenizer와 bert tokenizer의 호환성 문제 해결해야
def get_gpt_tokens(tokenizer, sentences, pretrained_model_name, max_len):
    input_ids = []
    attention_masks = []
    positional_ids = []

    return input_ids, attention_masks, positional_ids


def get_gpt_token_ids(tokenizer, sentences, max_len):
    outputs = tokenizer.encode_plus(sentences, add_special_tokens=True, max_length=max_len)
    input_ids = outputs['input_ids']
    attention_masks = outputs['attention_mask']
    positional_ids = outputs['positional_ids']

    return input_ids, attention_masks, positional_ids


def get_bert_tokens(tokenizer, sentences, labels, max_len):
    input_ids = []
    attention_masks = []
    positional_ids = []
    sequence_lengths = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

        tokens = tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0, :])
        positional_id, sequence_length = get_bilstm_positional_ids(max_len, tokens)
        positional_ids.append(positional_id)
        sequence_lengths.append(sequence_length)

    # must drop out after SEP of object and subject
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    positional_ids = torch.tensor(positional_ids)
    sequence_lengths = torch.tensor(sequence_lengths)

    return input_ids, attention_masks, positional_ids, sequence_lengths, labels


def add_type_lists_to_tokenizer(tokenizer):
    return tokenizer.add_tokens(type_list, special_tokens=True)


# %%

def getSubjStart(sentence, s):
    if sentence[s] in subject_type_list:
        return s
    return -1


def getSubjEnd(sentence, s):
    se = s
    if sentence[s] in subject_type_list:
        se += 1
    return se


def getObjStart(sentence, o):
    if sentence[o] in object_type_list:
        return o
    return -1


def getObjEnd(sentence, o):
    oe = o
    if sentence[o] in object_type_list:
        oe += 1
    return oe


def isEnd(i, length):
    if i + 1 >= length:
        return True
    return False


def getPE(max_length, sentence):
    length = len(sentence)
    ps = [0] * (length)
    po = [0] * (length)

    ss, se, os, oe = 0, 0, 0, 0

    flagS = False
    flagO = False
    for i in range(length):
        if sentence[i] == 'PAD':
            break

        if not isEnd(i, length):
            tmps = getSubjStart(sentence, i)
            tmpo = getObjStart(sentence, i)

            if tmps != -1 and not flagS:
                ss = tmps
                flagS = True
            elif tmps != -1 and flagS:
                tmp = getSubjEnd(sentence, i)
                if tmp > ss:
                    se = tmp

            if tmpo != -1 and not flagO:
                os = tmpo
                flagO = True
            elif tmpo != -1 and flagO:
                tmp = getObjEnd(sentence, i)
                if tmp > os:
                    oe = tmp

    real_length = 0
    firstSEP = True
    for i in range(len(sentence)):
        if i != 0 and real_length == i:
            break
        if sentence[i] == '[SEP]' and firstSEP:
            firstSEP = False
            real_length = i
            break
        elif sentence[i] == '[SEP]' and not firstSEP:
            break

        elif sentence[i] == '[PAD]' or sentence[i] == '[UKN]':
            ps[i] = 0
            po[i] = 0
            continue

        # -127 ... 0 ... 127
        # 1 ...   128 ... 256
        if i < ss:
            ps[i] = abs(i - ss)
        elif ss <= i <= se:
            ps[i] = max_length
        elif i > se:
            ps[i] = (i - se) + max_length

        if i < os:
            po[i] = abs(i - os)
        elif os <= i <= oe:
            po[i] = max_length
        elif i > oe:
            po[i] = (i - oe) + max_length
    # if real_length == 0:
    #   real_length = max_length
    return [ps, po], real_length


def get_bilstm_positional_ids(max_length, word_sequences):
    return getPE(max_length, word_sequences)
