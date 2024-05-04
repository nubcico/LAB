# THIS SCRIPT IS FOR GROUPING DATA BY SITES AND THEN CALCULATING THE F1 SCORE

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

mlb = MultiLabelBinarizer()


df2 = pd.read_json("trained_model\dl_project_test.json")


all_sites = []
for index, row in df2.iterrows():
    if row["source"] not in all_sites and row["large_label"] is not None:
        all_sites.append(row["source"])
print(len(all_sites))
print(all_sites)

with open("trained_model\cleaned_data2.txt", "r", encoding='utf-8') as f:
   lines = f.readlines()
unique_label = ''
i = 0

with open("trained_model\cleaned_data_sites3.txt", "a", encoding='utf-8') as f2:
    for line in lines:
        text_split = line.split('\t')
        true_split = text_split[0].split('-')
        pred_split = text_split[1].split('-')
        new_true = ''
        a = 0
        for word in true_split:
            # if a == 0:
            #    a = 1
            #    continue
            new_true += word.strip() + ' - '
        new_pred = ''
        for word in pred_split:
            new_pred += word.strip() + ' - '
        # print(new_true[:-3])
        if unique_label == '':
           unique_label = text_split[0].split('-')[0].strip() #unique label
        if unique_label != text_split[0].split('-')[0].strip():
           i += 1
           unique_label = text_split[0].split('-')[0].strip()
        row = str(all_sites[i]).strip() + '\t' + new_true[:-2].strip() + '\t' + text_split[1] #new_pred[:-1].strip() +'\n'

        f2.write(row)

df = pd.read_csv("trained_model\cleaned_data_sites2.txt", delimiter='\t', header=None, names=['source', 'label', 'preds'])

# Display the DataFrame
# print(df)
df['label'] = df['label'].str.split(' - ')
# print(df['label'])
df['preds'] = df['preds'].str.split(' - ')

# Group by the source and aggregate the label and preds columns into lists
grouped_data =  df.groupby('source')[['label', 'preds']].sum()



# data = list(set(grouped_data))
# print(data)
# print(grouped_data['label'])

# label_list = list(set(grouped_data['label'].sum()))
# preds_list = list(set(grouped_data['preds'].sum()))
# preds_list = list(set(grouped_data['preds'].explode()))
label_values = grouped_data['label'].explode().astype(str)
preds_values = grouped_data['preds'].explode().astype(str)

# Concatenate the two Series and convert to set
data = list(set(label_values.tolist() + preds_values.tolist()))


# data = list(set(grouped_data['label'].explode() + grouped_data['preds'].explode()))

print(data)
# print(preds_list)

# grouped_data = df.groupby('source')[['label', 'preds']].sum()

# label_values = grouped_data['label'].values
# preds_values = grouped_data['preds'].values

# # print(label_values)
# temp_set = set() 
# for label in label_values:
#     for word in label.split('- '):
#         temp_set.add(word.strip())
# label_list = list(temp_set)
# print(label_list)

# print(preds_values)

# preds_list = list(set(grouped_data['preds'].sum()))
# # data = list(set(grouped_data))
# print(label_list)
# print(preds_list)
# print(grouped_data)



# data = list(set(grouped_data))
# print(grouped_data["label"].tolist())
# print(data)
mlb.fit(data)
# print(mlb.classes_)

def to_single(x):
  a = mlb.transform(x) [0]
  for b in mlb.transform(x) [1:]:
    a = np.logical_or(a, b)
  return a
mrs = []
f1s = []

for t, p in zip(grouped_data['label'].tolist(),  grouped_data['preds'].tolist()):
# for t, p in zip(grouped_data['label'].explode().astype(str),  grouped_data['preds'].explode().astype(str)):
  y_pred = to_single(p).astype(np.int8)
  y_true = to_single(t).astype(np.int8)
  f1s.append(f1_score(y_true, y_pred))
  mrs.append(int(np.all(y_pred == y_true)))
avg_f1_score = np.mean(f1s)
avg_match_rate = np.mean(mrs)

print("Average F1 Score:", avg_f1_score)
print("Average Match Rate:", avg_match_rate)

# print(f1s)
# print(mrs)