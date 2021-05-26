import os
import re
import time
import pickle
import pandas as pd
import sentencepiece as spm
# Import Huggingface
from transformers import BertTokenizer
# Import custom modules
from utils import encoding_text

def preprocessing(args):

    start_time = time.time()

    print('Start preprocessing!')

    #===================================#
    #=============Data Load=============#
    #===================================#

    dataset_dict = dict()

    # 1-1) IMDB data open
    dataset_dict['imdb'] = {
        'train': pd.read_csv(os.path.join(args.imdb_data_path, 'train.csv'),
            names=['label', 'comment']).replace({
                'sentiment': {'positive': 0, 'negative': 1}
            }, inplace=True),
        'test': pd.read_csv(os.path.join(args.imdb_data_path, 'test.csv'),
            names=['label', 'comment']).replace({
                'sentiment': {'positive': 0, 'negative': 1}
            }, inplace=True))
    }

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

    print('Tokenizer setting...')

    # 1) Tokenizer open
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    #===================================#
    #=============Encoding==============#
    #===================================#

    print('Encoding...')
    for data in ['imdb', 'yelp', 'yahoo', 'ag_news', 'dbpedia']:
        train['comment'] = encoding_text(train['comment'], tokenizer, args.max_len)
        test['comment'] = encoding_text(test['comment'], tokenizer, args.max_len)

    #===================================#
    #==============Saving===============#
    #===================================#

    # 1) Print status
    print('Parsed sentence save setting...')

    max_train_len = max([len(x) for x in train['comment']])
    max_test_len = max([len(x) for x in test['comment']])
    mean_train_len = sum([len(x) for x in train['comment']]) / len(train['comment'])
    mean_test_len = sum([len(x) for x in test['comment']]) / len(test['comment'])

    print(f'Train data max length => comment: {max_train_len}')
    print(f'Train data mean length => comment: {mean_train_len}')
    print(f'Test data max length => comment: {max_test_len}')
    print(f'Test data mean length => comment: {mean_test_len}')

    # 2) Training pikcle saving
    with open(os.path.join(args.preprocess_path, 'processed.pkl'), 'wb') as f:
        pickle.dump({
            'train_comment_indices': train['comment'].tolist(),
            'test_comment_indices': test['comment'].tolist(),
            'train_label': train['sentiment'].tolist(),
            'test_label': test['sentiment'].tolist()
        }, f)

    print(f'Done! ; {round((time.time()-start_time)/60, 3)}min spend')