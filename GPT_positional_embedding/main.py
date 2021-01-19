import argparse
import pandas as pd
import numpy as np
import torch

from GPTDecoder import GPTDecoder
from Train import train
from Data import get_data

import random

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


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--mode", default=None, choices=["train", "valid", "eval", "finetune", "analysis", "feature_extraction"])

    parser.add_argument("--output_dir", type=str, default="dataset/models")
    parser.add_argument("--gpt2_version", type=str, default='gpt2-medium')
    parser.add_argument("--max_seq_length", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--do_preprocessing", type=bool, default=True)
    parser.add_argument("--train_file", type=str, default="dataset/json/train.json")
    parser.add_argument("--dev_file", type=str, default="dataset/json/dev.json")
    parser.add_argument("--test_file", type=str, default="dataset/json/test.json")

    args = parser.parse_args()

    output_dir = args.output_dir
    gpt2_version = args.gpt2_version
    max_len = args.max_seq_length
    batch_size = args.batch_size
    do_preprocessing = args.do_preprocessing
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file

    if do_preprocessing:
        get_data(train_file, 'train')
        get_data(dev_file, 'dev')
        get_data(test_file, 'test')

        train_file = 'dataset/train_masked.tsv'
        test_file = 'dataset/test_masked.tsv'
        dev_file = 'dataset/dev_masked.tsv'

    train_df = pd.read_csv(train_file, delimiter='\t', header=None,
                           names=['sentence_id', 'label', 'label_notes', 'sentence', 'masked_sentence'])
    dev_df = pd.read_csv(dev_file, delimiter='\t', header=None,
                         names=['sentence_id', 'label', 'label_notes', 'sentence', 'masked_sentence'])
    test_df = pd.read_csv(test_file, delimiter='\t', header=None,
                          names=['sentence_id', 'label', 'label_notes', 'sentence', 'masked_sentence'])

    train_sentences = np.array(train_df.sentence.values)
    train_masked_sentences = np.array(train_df.masked_sentence.values)
    train_labels = np.array([label_list.index(label) for label in train_df.label.values])

    dev_sentences = np.array(dev_df.sentence.values)
    dev_masked_sentences = np.array(dev_df.masked_sentence.values)
    dev_labels = np.array([label_list.index(label) for label in dev_df.label.values])

    test_sentences = np.array(test_df.sentence.values)
    test_masked_sentences = np.array(test_df.masked_sentence.values)
    test_labels = np.array([label_list.index(label) for label in test_df.label.values])

    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU avbailable, using the CPU instead')
        device = torch.device('cpu')

    train(device, train_sentences, train_masked_sentences, train_labels,
          dev_sentences, dev_masked_sentences, dev_labels,
          gpt2_version, max_len)


if __name__ == '__main__':
    main()
