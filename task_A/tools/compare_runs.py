import csv
import os

import pandas as pd

A_file = "introspection/introspection_<class 'task_A.frameworks.bert_framework_with_f.BERT_Framework_with_f'>_A0.830976_L0.604751.tsv"
B_file = "introspection/introspection_<class 'task_A.frameworks.bert_framework.BERT_Framework'>_A0.837710_L0.558764.tsv"

df_A = pd.read_csv(A_file, sep="\t").set_index('tweet_id')
df_B = pd.read_csv(B_file, sep="\t").set_index('tweet_id')


cols = \
[ 'tweet_id',
  'Correct_A',
  'Correct_B',
 'Ground truth',
 'Prediction_A',
 'Prediction_B',
 'Confidence_A',
 'Confidence_B',
 'branch_level',
 'Text',
 'Processed_Text']

with open(f"introspection/diff_{os.path.basename(A_file)}_{os.path.basename(B_file)}", "w") as csvf:
    csvwriter = csv.writer(csvf, delimiter='\t')
    csvwriter.writerow(cols)
    for index, A_row in df_A.iterrows():
        B_row = df_B.loc[index]
        if A_row["Correct"] != B_row["Correct"]:
            r = [
                index,
                A_row["Correct"],
                B_row["Correct"],
                A_row["Ground truth"],
                A_row["Prediction"],
                B_row["Prediction"],
                A_row["Confidence"],
                B_row["Confidence"],
                A_row["branch_level"],
                A_row["Text"],
                A_row["Processed_Text"]
            ]
            csvwriter.writerow(r)
