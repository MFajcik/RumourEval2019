"""
This is outer data_preprocessing file

To run:
    
python prep_pipeline.py

Main function has parameter that can be changed:

feats ('text' or 'SemEval')

"""
import json
import os

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm

from data_preprocessing.extract_thread_features import extract_thread_features_incl_response
from data_preprocessing.help_prep_functions import loadW2vModel
from data_preprocessing.preprocessing_reddit import load_data, load_test_data_reddit
from data_preprocessing.preprocessing_tweets import load_dataset, load_test_data_twitter
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


def prep_pipeline(dataset='RumEval2019', fset_name=None, feature_set=['avgw2v'], use_reddit_data=True):
    path = 'data_preprocessing/saved_data_' + dataset
    folds = {}
    # folds = load_dataset()
    #
    # if use_reddit_data:
    # reddit = load_data()
    #
    #     folds['train'].extend(reddit['train'])
    #     folds['dev'].extend(reddit['dev'])
    #     folds['test'].extend(reddit['test'])
    folds = load_test_data_twitter()
    reddit_data = load_test_data_reddit()
    folds['test'].extend(reddit_data['test'])
    #
    #     folds['train'].extend(reddit['train'])
    #     folds['dev'].extend(reddit['dev'])
    #     folds['test'].extend(reddit['test'])

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
                if not fset_name.endswith("test"):
                    fold_veracity_labels.append(convert_label(conversation['veracity']))
                conv_ids.append(conversation['id'])

        # % 0 supp, 1 comm,2 deny, 3 query
        if fold_features != []:
            path_fold = os.path.join(path, fold)
            print(f"Writing dataset {fold} for setting {fset_name}")
            if not os.path.exists(path_fold):
                os.makedirs(path_fold)
            if fset_name == "BUT_TEXT":
                jsonformat = {"Examples": []}
                cnt = 0
                already_known_tweetids = set()
                for fold_idx in tqdm(range(len(fold_features_dict))):
                    e = fold_features_dict[fold_idx]
                    tweet_ids_branch = tweet_ids[fold_idx]
                    branch_labels = fold_stance_labels[fold_idx].tolist()
                    for idx in range(len(e)):
                        if tweet_ids_branch[idx] in already_known_tweetids:
                            continue
                        else:
                            already_known_tweetids.add(tweet_ids_branch[idx])
                        print(f"{tweet_ids_branch[idx]} {branch_labels[idx]}")
                        example = {
                            "id": cnt,
                            "branch_id": f"{fold_idx}.{idx}",
                            "tweet_id": tweet_ids_branch[idx],
                            "stance_label": branch_labels[idx],
                            "veracity_label": fold_veracity_labels[fold_idx] if e[idx]["issource"] > 0 else -1,
                            "raw_text": e[idx]["raw_text"],
                            "raw_text_prev": e[idx - 1]["raw_text"] if idx - 1 > -1 else "",
                            "raw_text_src": e[0]["raw_text"] if idx - 1 > -1 else "",
                            "issource": e[idx]["issource"],
                            "spacy_processed_text": e[idx]["spacy_processed_text"],
                            'spacy_processed_BLvec': e[idx]['spacy_processed_BLvec'],
                            'spacy_processed_POSvec': e[idx]['spacy_processed_POSvec'],
                            'spacy_processed_DEPvec': e[idx]['spacy_processed_DEPvec'],
                            'spacy_processed_NERvec': e[idx]['spacy_processed_NERvec'],
                            "spacy_processed_text_prev": e[idx - 1]["spacy_processed_text"] if idx - 1 > -1 else "",
                            "spacy_processed_text_src": e[0]["spacy_processed_text"] if idx - 1 > -1 else ""
                        }
                        cnt += 1
                        example = {i: (v if type(v) is not np.ndarray else v.tolist())
                                   for i, v in example.items()}
                        jsonformat["Examples"].append(example)

                json.dump(jsonformat, open(os.path.join(path_fold, f"{fold}.json"), "w"))
            elif fset_name == "BUT_TEXT_test":
                jsonformat = {"Examples": []}
                cnt = 0
                already_known_tweetids = set()
                for fold_idx in tqdm(range(len(fold_features_dict))):
                    e = fold_features_dict[fold_idx]
                    tweet_ids_branch = tweet_ids[fold_idx]
                    branch_labels = fold_stance_labels[fold_idx].tolist()
                    for idx in range(len(e)):
                        if tweet_ids_branch[idx] in already_known_tweetids:
                            continue
                        else:
                            already_known_tweetids.add(tweet_ids_branch[idx])
                        example = {
                            "id": cnt,
                            "branch_id": f"{fold_idx}.{idx}",
                            "tweet_id": tweet_ids_branch[idx],
                            "stance_label": -1,
                            "veracity_label": -1,
                            "raw_text": e[idx]["raw_text"],
                            "raw_text_prev": e[idx - 1]["raw_text"] if idx - 1 > -1 else "",
                            "raw_text_src": e[0]["raw_text"] if idx - 1 > -1 else "",
                            "issource": e[idx]["issource"],
                            "spacy_processed_text": e[idx]["spacy_processed_text"],
                            'spacy_processed_BLvec': e[idx]['spacy_processed_BLvec'],
                            'spacy_processed_POSvec': e[idx]['spacy_processed_POSvec'],
                            'spacy_processed_DEPvec': e[idx]['spacy_processed_DEPvec'],
                            'spacy_processed_NERvec': e[idx]['spacy_processed_NERvec'],
                            "spacy_processed_text_prev": e[idx - 1]["spacy_processed_text"] if idx - 1 > -1 else "",
                            "spacy_processed_text_src": e[0]["spacy_processed_text"] if idx - 1 > -1 else ""
                        }
                        cnt += 1
                        example = {i: (v if type(v) is not np.ndarray else v.tolist())
                                   for i, v in example.items()}
                        jsonformat["Examples"].append(example)

                json.dump(jsonformat, open(os.path.join(path_fold, f"{fold}.json"), "w"))
            elif fset_name == "BUT_TEXT_VERACITY":
                jsonformat = {"Examples": []}
                cnt = 0
                already_known_tweetids = set()
                for fold_idx in tqdm(range(len(fold_features_dict))):
                    e = fold_features_dict[fold_idx]
                    tweet_ids_branch = tweet_ids[fold_idx]
                    branch_labels = fold_stance_labels[fold_idx].tolist()
                    for idx in range(len(e)):
                        if e[idx]["issource"] == 0:
                            continue
                        if tweet_ids_branch[idx] in already_known_tweetids:
                            continue
                        else:
                            already_known_tweetids.add(tweet_ids_branch[idx])
                        example = {
                            "id": cnt,
                            "branch_id": f"{fold_idx}.{idx}",
                            "tweet_id": tweet_ids_branch[idx],
                            "stance_label": branch_labels[idx],
                            "veracity_label": fold_veracity_labels[fold_idx],
                            "raw_text": e[idx]["raw_text"],
                            "raw_text_prev": e[idx - 1]["raw_text"] if idx - 1 > -1 else "",
                            "raw_text_src": e[0]["raw_text"] if idx - 1 > -1 else "",
                            "issource": e[idx]["issource"],
                            "spacy_processed_text": e[idx]["spacy_processed_text"],
                            'spacy_processed_BLvec': e[idx]['spacy_processed_BLvec'],
                            'spacy_processed_POSvec': e[idx]['spacy_processed_POSvec'],
                            'spacy_processed_DEPvec': e[idx]['spacy_processed_DEPvec'],
                            'spacy_processed_NERvec': e[idx]['spacy_processed_NERvec'],
                            "spacy_processed_text_prev": e[idx - 1]["spacy_processed_text"] if idx - 1 > -1 else "",
                            "spacy_processed_text_src": e[0]["spacy_processed_text"] if idx - 1 > -1 else ""
                        }
                        cnt += 1
                        example = {i: (v if type(v) is not np.ndarray else v.tolist())
                                   for i, v in example.items()}
                        jsonformat["Examples"].append(example)

                json.dump(jsonformat, open(os.path.join(path_fold, f"{fold}.json"), "w"))

            elif fset_name == "BUTFeatures_Branch":
                jsonformat = {"Examples": []}
                alltexts = []
                for i, e in enumerate(range(len(fold_features_dict))):
                    for j in range(len(fold_features_dict[i])):
                        fold_features_dict[i][j]["avgw2v"] = fold_features_dict[i][j]["avgw2v"].tolist()
                        if fold_features_dict[i][j]["raw_text"] not in alltexts:
                            alltexts.append(fold_features_dict[i][j]["raw_text"])
                        fold_features_dict[i][j]["string_id"] = alltexts.index(fold_features_dict[i][j]["raw_text"])
                    example = {
                        "id": i,
                        "stance_labels": fold_stance_labels[i].tolist(),
                        "veracity_label": fold_veracity_labels[i],
                        "features": fold_features_dict[i]
                    }
                    jsonformat["Examples"].append(example)
                json.dump(jsonformat, open(os.path.join(path_fold, f"{fold}.json"), "w"))
            elif fset_name == "BUT_Features":
                jsonformat = {"Examples": []}
                cnt = 0
                already_known_tweetids = set()
                for fold_idx in tqdm(range(len(fold_features_dict))):
                    e = fold_features_dict[fold_idx]
                    tweet_ids_branch = tweet_ids[fold_idx]
                    for idx in range(len(e)):
                        if tweet_ids_branch[idx] in already_known_tweetids:
                            continue
                        else:
                            already_known_tweetids.add(tweet_ids_branch[idx])
                            exampleAdditional = {
                                "id": cnt,
                                "branch_id": f"{fold_idx}.{idx}",
                                "tweet_id": tweet_ids_branch[idx],
                                "stance_label": fold_stance_labels[fold_idx].tolist()[idx],
                                "veracity_label": fold_veracity_labels[fold_idx] if e[idx]["issource"] > 0 else -1,
                                "raw_text_prev": e[idx - 1]["raw_text"] if idx - 1 > -1 else "",
                                "raw_text_src": e[0]["raw_text"] if idx - 1 > -1 else "",
                                "spacy_processed_text_prev": e[idx - 1]["spacy_processed_text"] if idx - 1 > -1 else "",
                                "spacy_processed_text_src": e[0]["spacy_processed_text"] if idx - 1 > -1 else ""
                            }
                            exampleOriginal = {i: (v if type(v) is not np.ndarray else v.tolist())
                                               for i, v in e[idx].items()}
                            example = {**exampleOriginal, **exampleAdditional}
                            cnt += 1
                            jsonformat["Examples"].append(example)
                json.dump(jsonformat, open(os.path.join(path_fold, f"{fold}.json"), "w"))
            else:
                fold_features = pad_sequences(fold_features, maxlen=None,
                                              dtype='float32',
                                              padding='post',
                                              truncating='post', value=0.)

                fold_stance_labels = pad_sequences(fold_stance_labels, maxlen=None,
                                                   dtype='float32',
                                                   padding='post', truncating='post',
                                                   value=0.)

                fold_veracity_labels = np.asarray(fold_veracity_labels)

                np.save(os.path.join(path_fold, 'train_array'), fold_features)
                np.save(os.path.join(path_fold, 'labels'), fold_veracity_labels)
                np.save(os.path.join(path_fold, 'fold_stance_labels'),
                        fold_stance_labels)
                np.save(os.path.join(path_fold, 'ids'), conv_ids)
                np.save(os.path.join(path_fold, 'tweet_ids'), tweet_ids)


# %%
def main(feats="BUT_TEXT_test"):
    if feats == 'SemEvalfeatures':
        SemEvalfeatures = ['avgw2v', 'hasnegation', 'hasswearwords',
                           'capitalratio', 'hasperiod', 'hasqmark',
                           'hasemark', 'hasurl', 'haspic',
                           'charcount', 'wordcount', 'issource',
                           'Word2VecSimilarityWrtOther',
                           'Word2VecSimilarityWrtSource',
                           'Word2VecSimilarityWrtPrev']
        prep_pipeline(dataset='RumEval2019', fset_name=feats, feature_set=SemEvalfeatures)
    elif feats == "BUTFeatures_Branch":
        features = ['avgw2v',
                    'hasnegation', 'hasswearwords',
                    'capitalratio', 'hasperiod', 'hasqmark',
                    'hasemark', 'hasurl', 'haspic',
                    'charcount', 'wordcount', 'issource',
                    'Word2VecSimilarityWrtOther',
                    'Word2VecSimilarityWrtSource',
                    'Word2VecSimilarityWrtPrev',
                    'issource',
                    'raw_text',
                    'spacy_processed_text']
        prep_pipeline(dataset='RumEval2019', fset_name=feats, feature_set=features)
    elif feats == "BUT_TEXT":
        features = [
            'issource',
            'raw_text',
            'spacy_processed_text',
            'spacy_processed_BLvec',
            'spacy_processed_POSvec',
            'spacy_processed_DEPvec',
            'spacy_processed_NERvec']
        prep_pipeline(dataset='RumEval2019', fset_name=feats, feature_set=features)
    elif feats == "BUT_TEXT_test":
        features = [
            'issource',
            'raw_text',
            'spacy_processed_text',
            'spacy_processed_BLvec',
            'spacy_processed_POSvec',
            'spacy_processed_DEPvec',
            'spacy_processed_NERvec']
        prep_pipeline(dataset='RumEval2019', fset_name=feats, feature_set=features)
    elif feats == "BUT_TEXT_VERACITY":
        features = [
            'issource',
            'raw_text',
            'spacy_processed_text',
            'spacy_processed_BLvec',
            'spacy_processed_POSvec',
            'spacy_processed_DEPvec',
            'spacy_processed_NERvec']
        prep_pipeline(dataset='RumEval2019', fset_name=feats, feature_set=features)
    elif feats == "BUT_Features":
        features = [
            'avgw2v',
            'hasnegation', 'hasswearwords',
            'capitalratio', 'hasperiod', 'hasqmark',
            'hasemark', 'hasurl', 'haspic',
            'charcount', 'wordcount', 'issource',
            'Word2VecSimilarityWrtOther',
            'Word2VecSimilarityWrtSource',
            'Word2VecSimilarityWrtPrev',
            'issource',
            'raw_text',
            'spacy_processed_text',
            'spacy_processed_BLvec',
            'spacy_processed_POSvec',
            'spacy_processed_DEPvec',
            'spacy_processed_NERvec',
            'src_num_false_synonyms',
            'src_num_false_antonyms',
            'thread_num_false_synonyms',
            'thread_num_false_antonyms',
            'src_unconfirmed',
            'src_rumour',
            'thread_unconfirmed',
            'thread_rumour',
            'src_num_wh',
            'thread_num_wh']
        prep_pipeline(dataset='RumEval2019', fset_name=feats, feature_set=features)


if __name__ == '__main__':
    main()
