"""
This is outer data_preprocessing file

To run:
    
python prep_pipeline.py

Main function has parameter that can be changed:

feats ('text' or 'SemEval')

"""
import json
import os

from tqdm import tqdm

from data_preprocessing.extract_thread_features import extract_thread_features_incl_response
from data_preprocessing.help_prep_functions import loadW2vModel
from data_preprocessing.preprocessing_reddit import load_data
from data_preprocessing.preprocessing_tweets import load_dataset
from data_preprocessing.transform_feature_dict import transform_feature_dict


# import numpy as np
# from keras.data_preprocessing.sequence import pad_sequences

# %%

def convert_label(label):
    if label == "true":
        return (0)
    elif label == "false":
        return (1)
    elif label == "unverified":
        return (2)
    else:
        print(label)


def prep_pipeline(dataset='RumEval2019', feature_set=['avgw2v'], use_reddit_data=True):
    path = 'data_preprocessing/saved_data_' + dataset
    folds = {}
    folds = load_dataset()

    if use_reddit_data:
        reddit = load_data()

        folds['train'].extend(reddit['train'])
        folds['dev'].extend(reddit['dev'])
        folds['test'].extend(reddit['test'])

    loadW2vModel()

    # %%
    # data folds , i.e. train, dev, test
    for fold in folds.keys():

        print(fold)
        # contains features for each branch in all conversations
        # shape shape conversations_count *BRANCH_COUNT x BRANCH_LEN x FEATURE vector
        fold_features = []
        fold_features_dict = []
        # contains ids of processed tweets in branches in all conversations
        #  shape conversations_count * BRANCH_COUNT x BRANCH_len x String
        tweet_ids = []
        # contains stance labels for all branches in all conversations
        # final shape conversations_count * BRANCH_COUNT for the conversation x BRANCH_len
        fold_stance_labels = []
        fold_veracity_labels = []
        conv_ids = []

        all_fold_features = []
        for conversation in tqdm(folds[fold]):
            # extract features for source and replies
            thread_feature_dict = extract_thread_features_incl_response(conversation)
            all_fold_features.append(thread_feature_dict)

            thread_features_array, thread_features_dict, thread_stance_labels, branches = transform_feature_dict(
                thread_feature_dict, conversation,
                feature_set=feature_set)

            fold_features_dict.extend(thread_features_dict)
            fold_stance_labels.extend(thread_stance_labels)
            tweet_ids.extend(branches)
            fold_features.extend(thread_features_array)

            # build data for source tweet for veracity
            for i in range(len(thread_features_array)):
                fold_veracity_labels.append(convert_label(conversation['veracity']))
                conv_ids.append(conversation['id'])

        # %
        if fold_features != []:

            path_fold = os.path.join(path, fold)
            if not os.path.exists(path_fold):
                os.makedirs(path_fold)
            jsonformat = {"Examples": []}
            for i, e in enumerate(range(len(fold_features_dict))):
                for j in range(len(fold_features_dict[i])):
                    fold_features_dict[i][j]["avgw2v"] = fold_features_dict[i][j]["avgw2v"].tolist()
                example = {
                    "id": i,
                    "stance_labels": fold_stance_labels[i].tolist(),
                    "veracity_label": fold_veracity_labels[i],
                    "features": fold_features_dict[i]
                }
                jsonformat["Examples"].append(example)
            json.dump(jsonformat, open(os.path.join(path_fold, f"{fold}.json"), "w"))

            # fold_features = pad_sequences(fold_features, maxlen=None,
            #                               dtype='float32',
            #                               padding='post',
            #                               truncating='post', value=0.)
            #
            # fold_stance_labels = pad_sequences(fold_stance_labels, maxlen=None,
            #                                    dtype='float32',
            #                                    padding='post', truncating='post',
            #                                    value=0.)
            #
            # fold_veracity_labels = np.asarray(fold_veracity_labels)
            #
            # np.save(os.path.join(path_fold, 'train_array'), fold_features)
            # np.save(os.path.join(path_fold, 'labels'), fold_veracity_labels)
            # np.save(os.path.join(path_fold, 'fold_stance_labels'),
            #         fold_stance_labels)
            # np.save(os.path.join(path_fold, 'ids'), conv_ids)
            # np.save(os.path.join(path_fold, 'tweet_ids'), tweet_ids)


# %%
def main(data='RumEval2019', feats='BUTFeatures'):
    if feats == 'text':
        prep_pipeline(dataset='RumEval2019', feature_set=['avgw2v'])
    elif feats == 'SemEvalfeatures':
        SemEvalfeatures = ['avgw2v', 'hasnegation', 'hasswearwords',
                           'capitalratio', 'hasperiod', 'hasqmark',
                           'hasemark', 'hasurl', 'haspic',
                           'charcount', 'wordcount', 'issource',
                           'Word2VecSimilarityWrtOther',
                           'Word2VecSimilarityWrtSource',
                           'Word2VecSimilarityWrtPrev']
        prep_pipeline(dataset='RumEval2019', feature_set=SemEvalfeatures)
    elif feats == "BUTFeatures":
        features = ['avgw2v', 'hasnegation', 'hasswearwords',
                    'capitalratio', 'hasperiod', 'hasqmark',
                    'hasemark', 'hasurl', 'haspic',
                    'charcount', 'wordcount', 'issource',
                    'Word2VecSimilarityWrtOther',
                    'Word2VecSimilarityWrtSource',
                    'Word2VecSimilarityWrtPrev',
                    'raw_text',
                    'spacy_processed_text']
        prep_pipeline(dataset='RumEval2019', feature_set=features)


if __name__ == '__main__':
    main()
