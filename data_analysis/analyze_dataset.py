import json
import numpy as np

from task_A.frameworks.bert_framework import map_s_to_label_stance


def analyze_BRANCH_DATASET(fpath):
    with open(fpath) as dataf:
        data_json = json.load(dataf)
        total_examples = 0
        total_source = 0
        total_ids = []
        for e in data_json["Examples"]:
            for k in range(len(e["stance_labels"])):
                total_examples += 1
                if e["features"][k]["issource"] == 1:
                    total_source += 1
                if e["features"][k]["string_id"] not in total_ids:
                    total_ids.append(e["features"][k]["string_id"])
        print(f"Total examples: {total_examples}")
        print(f"Total unique examples: {len(total_ids)}")
        print(f"Total source: {total_source}")


def analyze_BERT_DATASET(fpath):
    examples = 0
    with open(fpath) as dataf:
        data_json = json.load(dataf)
        classes = [0] * 4
        source_classes = [0] * 4
        for e in data_json["Examples"]:
            examples += 1
            classes[e["stance_label"]] += 1
            if e["issource"] == 1:
                source_classes[e["stance_label"]] += 1
    print(f"Total unique examples: {examples}")
    print_clsinfo(classes)
    print_clsinfo(source_classes)


def print_clsinfo(classes):
    print(f"Class balance: {classes}")
    classes = np.array(classes)
    print(f"Class balance [ normalized ]: {(classes / classes.sum() * 100)}")

def analyze_answers(fpath):
    examples = 0
    with open(fpath) as dataf:
        data_json = json.load(dataf)
        classes = [0] * 4
        for e in data_json["subtaskaenglish"]:
            examples += 1
            cls = map_s_to_label_stance[data_json["subtaskaenglish"][e]]
            classes[cls] += 1

    print(f"Total unique examples: {examples}")
    print_clsinfo(classes)


# classes
# Support / Deny / Query / Comment

# print("BRANCH DATASET")
# analyze_BRANCH_DATASET("data_preprocessing/saved_data_RumEval2019_BRANCH/train/train.json")

print (map_s_to_label_stance.keys())
print("BERT DATASET")
print("TRAIN")
analyze_BERT_DATASET("data_preprocessing/saved_data_RumEval2019_SEQ/train/train.json")
print("DEV")
analyze_BERT_DATASET("data_preprocessing/saved_data_RumEval2019_SEQ/dev/dev.json")


print("MODEL_PREDS")
analyze_answers("answer_BERT_textnsource.json")
