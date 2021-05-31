import os
import re
import time
import emoji
import pickle
import logging
import pandas as pd
import sentencepiece as spm
# Import Huggingface
from transformers import BertTokenizer
# Import custom modules
from utils import encoding_text, TqdmLoggingHandler, write_log

def encoding_text(list_x, tokenizer, max_len):

    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(r'<[^>]+>')

    def clean(x):
        x = pattern.sub(' ', x)
        x = x.strip()
        return x

    encoded_text_list = list_x.map(lambda x: tokenizer.encode(
        clean(str(x)),
        max_length=max_len,
        truncation=True
    ))
    return encoded_text_list

def preprocessing(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, 'Start preprocessing!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    dataset_dict = dict()

    # 1-1) IMDB data open
    dataset_dict['imdb'] = {
        'train': pd.read_csv(os.path.join(args.imdb_data_path, 'train.csv'),
            names=['label', 'comment']),
        'test': pd.read_csv(os.path.join(args.imdb_data_path, 'test.csv'),
            names=['label', 'comment'])
    }
    dataset_dict['imdb']['train'].replace({
                'sentiment': {'positive': 0, 'negative': 1}
            }, inplace=True)
    dataset_dict['imdb']['test'].replace({
                'sentiment': {'positive': 0, 'negative': 1}
            }, inplace=True)

    # 1-2) Yelp data open
    dataset_dict['yelp'] = {
        'train': pd.read_csv(os.path.join(args.yelp_data_path, 'train.csv'), 
            names=['label', 'comment']),
        'test': pd.read_csv(os.path.join(args.yelp_data_path, 'test.csv'), 
            names=['label', 'comment'])
    }

    # 1-3) Yahoo data open
    dataset_dict['yahoo'] = {
        'train': pd.read_csv(os.path.join(args.yahoo_data_path, 'train.csv'), 
            names=['label', 'question_title', 'question_content', 'best_answer']),
        'test': pd.read_csv(os.path.join(args.yahoo_data_path, 'test.csv'), 
            names=['label', 'question_title', 'question_content', 'best_answer']),
    }

    # 1-4) AG News data open
    dataset_dict['ag_news'] = {
        'train': pd.read_csv(os.path.join(args.ag_news_data_path, 'train.csv'), 
            names=['label', 'title', 'description']),
        'test': pd.read_csv(os.path.join(args.ag_news_data_path, 'test.csv'), 
            names=['label', 'title', 'description'])
    }

    # 1-5) DBpedia data open
    dataset_dict['dbpedia'] = {
        'train': pd.read_csv(os.path.join(args.dbpedia_data_path, 'train.csv'), 
            names=['label', 'title', 'description']),
        'test': pd.read_csv(os.path.join(args.dbpedia_data_path, 'test.csv'), 
            names=['label', 'title', 'description'])
    }

    # 2) Path setting
    if not os.path.exists(args.preprocess_path):
        os.mkdir(args.preprocess_path)

    #===================================#
    #=============Tokenizer=============#
    #===================================#

    write_log(logger, 'Tokenizer setting...')

    # 1) Tokenizer open
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    #===================================#
    #=============Encoding==============#
    #===================================#

    for data in ['imdb', 'yelp', 'yahoo', 'ag_news', 'dbpedia']:
        write_log(logger, f'Encoding {data}...')
        col_list = list(dataset_dict[data]['train'].columns)
        col_list.remove('label')
        for col in col_list:
            dataset_dict[data]['train'][col] = \
                encoding_text(dataset_dict[data]['train'][col], tokenizer, args.max_len)
            dataset_dict[data]['test'][col] = \
                encoding_text(dataset_dict[data]['test'][col], tokenizer, args.max_len)

    #===================================#
    #==============Saving===============#
    #===================================#

    # 1) Print status
    write_log(logger, 'Parsed sentence save setting...')

    for data in ['imdb', 'yelp', 'yahoo', 'ag_news', 'dbpedia']:

        col_list = list(dataset_dict[data]['train'].columns)
        col_list.remove('label')

        max_train_len, max_test_len = list(), list()
        mean_train_len, mean_test_len = list(), list()

        train_len = len(dataset_dict[data]['train'])
        test_len = len(dataset_dict[data]['test'])

        for col in col_list:
            max_train_len.append(max([len(x) for x in dataset_dict[data]['train'][col]]))
            max_test_len.append(max([len(x) for x in dataset_dict[data]['test'][col]]))
            mean_train_len.append(sum([len(x) for x in dataset_dict[data]['train'][col]]) / train_len)
            mean_test_len.append(sum([len(x) for x in dataset_dict[data]['test'][col]]) / test_len)
        max_train_len = max(max_train_len)
        max_test_len = max(max_test_len)
        mean_train_len = sum(mean_train_len) / len(col_list)
        mean_test_len = sum(mean_test_len) / len(col_list)

        print(f'--- {data} Dataset ---')
        print(f'Train data max length => comment: {max_train_len}')
        print(f'Train data mean length => comment: {mean_train_len}')
        print(f'Test data max length => comment: {max_test_len}')
        print(f'Test data mean length => comment: {mean_test_len}')
        print()

    # 2) Training pikcle saving
    with open(os.path.join(args.preprocess_path, 'augmented_processed.pkl'), 'wb') as f:
        pickle.dump(dataset_dict, f)

    write_log(logger, f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')