"""
This code helps convert dictionaries of features from conversation into
arrays of branches of conversation
"""

import numpy as np


# %%
def convert_label(label):
    if label == "support":
        return (0)
    elif label == "comment":
        return (1)
    elif label == "deny":
        return (2)
    elif label == "query":
        return (3)
    else:
        print(label)


# %%
def transform_feature_dict(thread_feature_dict, conversation, feature_set):
    """
    :return:
    thread_features_array -  contains features for each branch in conversation
                             shape BRANCH_COUNT x BRANCH_LEN x FEATURE vector
    thread_stance_labels -  contains labels (1 dimensional) for each branch in conversation
                             shape BRANCH_COUNT x BRANCH_LEN x 1
    clean_branches - contains ids of processed tweets in branches
                     shape BRANCH_COUNT x BRANCH_len x String
    """
    thread_features_array = []
    thread_features_dict = []
    thread_stance_labels = []
    clean_branches = []

    branches = conversation['branches']

    for branch in branches:
        branch_rep = []
        branch_rep_dicts = []
        # contains ids for tweets
        clb = []
        branch_stance_lab = []
        for twid in branch:
            if twid in thread_feature_dict.keys():
                tweet_rep, tweet_rep_dict = dict_to_array_and_dict(thread_feature_dict[twid], feature_set)
                branch_rep.append(tweet_rep)
                branch_rep_dicts.append(tweet_rep_dict)

                # if it is source tweet
                if twid == branch[0]:
                    # if it is labelled
                    if 'label' in list(conversation['source'].keys()):
                        branch_stance_lab.append(convert_label(
                            conversation['source']['label']))
                    clb.append(twid)
                else:
                    for r in conversation['replies']:
                        if r['id_str'] == twid:
                            if 'label' in list(r.keys()):
                                branch_stance_lab.append(
                                    convert_label(r['label']))
                            clb.append(twid)
        if branch_rep != []:
            branch_rep = np.asarray(branch_rep)
            branch_stance_lab = np.asarray(branch_stance_lab)
            thread_features_array.append(branch_rep)
            thread_features_dict.append(branch_rep_dicts)
            thread_stance_labels.append(branch_stance_lab)
            clean_branches.append(clb)

    return thread_features_array, thread_features_dict, thread_stance_labels, clean_branches


# %%
def dict_to_array(feature_dict, feature_set):
    """
    Create array from selected features
    :param feature_dict:
    :param feature_set:
    :return:
    """
    tweet_rep = []
    for feature_name in feature_set:

        if np.isscalar(feature_dict[feature_name]):
            tweet_rep.append(feature_dict[feature_name])
        else:
            tweet_rep.extend(feature_dict[feature_name])
    tweet_rep = np.asarray(tweet_rep)
    return tweet_rep


def dict_to_array_and_dict(feature_dict, feature_set):
    """
    Create array from selected features
    :param feature_dict:
    :param feature_set:
    :return:
    """
    tweet_rep = []
    tweet_rep_d = dict()
    for feature_name in feature_set:
        tweet_rep_d[feature_name] = feature_dict[feature_name]
        if np.isscalar(feature_dict[feature_name]):
            tweet_rep.append(feature_dict[feature_name])
        else:
            tweet_rep.extend(feature_dict[feature_name])

    tweet_rep = np.asarray(tweet_rep)
    return tweet_rep, tweet_rep_d
