# -*- coding: utf-8 -*-
"""
This file contains function to load tweets

"""
import json
import os

from data_preprocessing.tree2branches import tree2branches

# %%
TRAIN_DATA_PREFIX = "/home/ifajcik/Work/NLP/semeval_2019/7_Rumour_Eval/rumoureval-2019-training-data"


def load_true_labels():
    tweet_label_dict = {}
    veracity_label_dict = {}
    path_dev = os.path.join(TRAIN_DATA_PREFIX, "dev-key.json")
    with open(path_dev, 'r') as f:
        dev_key = json.load(f)

    path_train = os.path.join(TRAIN_DATA_PREFIX, "train-key.json")
    with open(path_train, 'r') as f:
        train_key = json.load(f)

    tweet_label_dict['dev'] = dev_key['subtaskaenglish']
    tweet_label_dict['train'] = train_key['subtaskaenglish']

    veracity_label_dict['dev'] = dev_key['subtaskbenglish']
    veracity_label_dict['train'] = train_key['subtaskbenglish']

    return tweet_label_dict, veracity_label_dict

def load_test_data_twitter(path ="/home/ifajcik/Work/NLP/semeval_2019/7_Rumour_Eval/dummy_test_data/twitter-english"):
    allconv = []
    train_dev_split = {}
    train_dev_split['dev'] = []
    train_dev_split['train'] = []
    train_dev_split['test'] = []
    tweet_data = sorted(os.listdir(path))
    newfolds = [i for i in tweet_data if i[0] != '.']
    tweet_data = newfolds # conversation ids, source post id == conversation id
    conversation = {}
    # build conversations for tweet group
    for foldr in tweet_data:
        flag = 0
        conversation['id'] = foldr
        tweets = []
        path_repl = path + '/' + foldr + '/replies'
        files_t = sorted(os.listdir(path_repl))
        newfolds = [i for i in files_t if i[0] != '.']
        files_t = newfolds
        flag = "test"
        if files_t != []:
            # iterate over json reply files
            for repl_file in files_t:
                with open(os.path.join(path_repl, repl_file)) as f:
                    for line in f:
                        tw = json.loads(line)
                        tw['used'] = 0

                        tw['set'] = flag
                        tweets.append(tw)
                        if tw['text'] is None:
                            print("Tweet has no text", tw['id'])
            conversation['replies'] = tweets

            path_src = path + '/' + foldr + '/source-tweet'
            files_t = sorted(os.listdir(path_src))
            with open(os.path.join(path_src, files_t[0])) as f:
                for line in f:
                    src = json.loads(line)
                    src['used'] = 0
                    scrcid = src['id_str']
                    src['set'] = flag

            conversation['source'] = src
            if src['text'] is None:
                print("Tweet has no text", src['id'])
            path_struct = path + '/' + foldr + '/structure.json'
            with open(path_struct) as f:
                for line in f:
                    struct = json.loads(line)
            if len(struct) > 1:
                new_struct = {}
                new_struct[foldr] = struct[foldr]
                struct = new_struct
                # Take item from structure if key is same as source tweet id
            conversation['structure'] = struct

            branches = tree2branches(conversation['structure'])
            conversation['branches'] = branches
            train_dev_split[flag].append(conversation.copy())
            allconv.append(conversation.copy())

        # if no replies are present, still add just source
        else:
            flag = 'test'
            path_src = path + '/' + foldr + '/source-tweet'
            files_t = sorted(os.listdir(path_src))
            with open(os.path.join(path_src, files_t[0])) as f:
                for line in f:
                    src = json.loads(line)
                    src['used'] = 0
                    scrcid = src['id_str']
                    src['set'] = flag

            conversation['source'] = src
            if src['text'] is None:
                print("Tweet has no text", src['id'])

            path_struct = path + '/' + foldr + '/structure.json'
            with open(path_struct) as f:
                for line in f:
                    struct = json.loads(line)
            if len(struct) > 1:
                # print "Structure has more than one root"
                new_struct = {}
                new_struct[foldr] = struct[foldr]
                struct = new_struct
                # Take item from structure if key is same as source tweet id
            conversation['structure'] = struct
            branches = tree2branches(conversation['structure'])

            conversation['branches'] = branches
            train_dev_split[flag].append(conversation.copy())
            allconv.append(conversation.copy())

            print(foldr)

    allconv = []

    return train_dev_split

# %%
def load_dataset():
    # Load labels and split for task A and task B
    tweet_label_dict, veracity_label_dict = load_true_labels()
    dev = tweet_label_dict['dev']
    train = tweet_label_dict['train']
    dev_tweets = dev.keys()
    train_tweets = train.keys()
    # Load folds and conversations
    path_to_folds = os.path.join(TRAIN_DATA_PREFIX, 'twitter-english')
    folds = sorted(os.listdir(path_to_folds))
    newfolds = [i for i in folds if i[0] != '.']
    folds = newfolds
    cvfolds = {}
    allconv = []
    train_dev_split = {}
    train_dev_split['dev'] = []
    train_dev_split['train'] = []
    train_dev_split['test'] = []
    # iterate over all tweet groups [reffered to as 'folds'] - charliehebdo etc...
    for nfold, fold in enumerate(folds):
        path_to_tweets = os.path.join(path_to_folds, fold)
        tweet_data = sorted(os.listdir(path_to_tweets))
        newfolds = [i for i in tweet_data if i[0] != '.']
        tweet_data = newfolds # conversation ids, source post id == conversation id
        conversation = {}
        # build conversations for tweet group
        for foldr in tweet_data:
            flag = 0
            conversation['id'] = foldr
            tweets = []
            path_repl = path_to_tweets + '/' + foldr + '/replies'
            files_t = sorted(os.listdir(path_repl))
            newfolds = [i for i in files_t if i[0] != '.']
            files_t = newfolds
            if files_t != []:
                # iterate over json reply files
                for repl_file in files_t:
                    with open(os.path.join(path_repl, repl_file)) as f:
                        for line in f:
                            tw = json.loads(line)
                            tw['used'] = 0
                            replyid = tw['id_str']

                            # check if tweet belongs to dev fold
                            if replyid in dev_tweets:
                                tw['set'] = 'dev'
                                tw['label'] = dev[replyid]
                                #                        train_dev_tweets['dev'].append(tw)
                                if flag == 'train':
                                    print("The tree is split between sets", foldr)
                                flag = 'dev'
                            elif replyid in train_tweets:
                                tw['set'] = 'train'
                                tw['label'] = train[replyid]
                                #                        train_dev_tweets['train'].append(tw)
                                if flag == 'dev':
                                    print("The tree is split between sets", foldr)
                                flag = 'train'
                            else:
                                print("Tweet was not found! ID: ", foldr)
                            tweets.append(tw)
                            if tw['text'] is None:
                                print("Tweet has no text", tw['id'])
                conversation['replies'] = tweets

                path_src = path_to_tweets + '/' + foldr + '/source-tweet'
                files_t = sorted(os.listdir(path_src))
                with open(os.path.join(path_src, files_t[0])) as f:
                    for line in f:
                        src = json.loads(line)
                        src['used'] = 0
                        scrcid = src['id_str']
                        src['set'] = flag
                        src['label'] = tweet_label_dict[flag][scrcid]

                conversation['source'] = src
                conversation['veracity'] = veracity_label_dict[flag][scrcid]
                if src['text'] is None:
                    print("Tweet has no text", src['id'])
                path_struct = path_to_tweets + '/' + foldr + '/structure.json'
                with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)
                if len(struct) > 1:
                    # I had to alter the structure of this conversation
                    if foldr == '553480082996879360':
                        new_struct = {}
                        new_struct[foldr] = struct[foldr]
                        new_struct[foldr]['553495625527209985'] = struct['553485679129534464']['553495625527209985']
                        new_struct[foldr]['553495937432432640'] = struct['553490097623269376']['553495937432432640']
                        struct = new_struct
                    else:
                        new_struct = {}
                        new_struct[foldr] = struct[foldr]
                        struct = new_struct
                    # Take item from structure if key is same as source tweet id
                conversation['structure'] = struct

                branches = tree2branches(conversation['structure'])
                conversation['branches'] = branches
                train_dev_split[flag].append(conversation.copy())
                allconv.append(conversation.copy())
            else:
                flag = 'train'
                path_src = path_to_tweets + '/' + foldr + '/source-tweet'
                files_t = sorted(os.listdir(path_src))
                with open(os.path.join(path_src, files_t[0])) as f:
                    for line in f:
                        src = json.loads(line)
                        src['used'] = 0
                        scrcid = src['id_str']
                        src['set'] = flag
                        src['label'] = tweet_label_dict[flag][scrcid]

                conversation['source'] = src
                conversation['veracity'] = veracity_label_dict[flag][scrcid]
                if src['text'] is None:
                    print("Tweet has no text", src['id'])

                path_struct = path_to_tweets + '/' + foldr + '/structure.json'
                with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)
                if len(struct) > 1:
                    # print "Structure has more than one root"
                    new_struct = {}
                    new_struct[foldr] = struct[foldr]
                    struct = new_struct
                    # Take item from structure if key is same as source tweet id
                conversation['structure'] = struct
                branches = tree2branches(conversation['structure'])

                conversation['branches'] = branches
                train_dev_split[flag].append(conversation.copy())
                allconv.append(conversation.copy())

                print(foldr)

        cvfolds[fold] = allconv
        allconv = []

    return train_dev_split
