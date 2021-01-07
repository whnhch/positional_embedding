import argparse
import pandas as pd
import numpy as np

from GPTDecoder import GPTDecoder
from train import train
from data import get_data

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
    parser.add_argument("--pretrained_model_name", type=str, default='gpt-medium')
    parser.add_argument("--max_seq_length", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--do_preprocessing", type=bool, default=True)
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument("--dev_file", type=str, default="dev.json")
    parser.add_argument("--test_file", type=str, default="test.json")
    args = parser.parse_args()

    output_dir = args.output_dir
    pretrained_model_name = args.pretrained_model_name
    max_len = args.max_seq_length
    batch_size = args.batch_size
    do_preprocessing = args.do_preprocessing
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file

    if do_preprocessing:
        train_df = get_data(train_file)
        dev_df = get_data(dev_file)
        test_df = get_data(test_file)
    else:
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

    train(train_sentences, train_masked_sentences, train_labels,
          dev_sentences, dev_masked_sentences, dev_labels,
          pretrained_model_name, max_len)

    # GDPDecoder Result
    decoder = GPTDecoder.GPTDecoder(pretrained_model_name, max_len)
    decoder()
    train_sentences
