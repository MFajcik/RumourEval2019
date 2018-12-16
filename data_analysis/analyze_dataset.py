import json

with open("data_preprocessing/saved_data_RumEval2019_WITH_PATCHES/train/train.json") as dataf:
    data_json = json.load(dataf)
    total_examples = 0
    total_source = 0
    for e in data_json["Examples"]:
        for k in range(len(e["stance_labels"])):
            total_examples += 1
            if e["features"][k]["issource"] == 1:
                total_source += 1
    print(f"Total examples: {total_examples}")
    print(f"Total source: {total_source}")
