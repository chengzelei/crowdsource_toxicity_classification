import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/questions_labeled/train_feihuo.csv')
response = df['question']
labels_feihuo = df['label'].to_numpy()
num_labels = len(labels_feihuo)
labels_feihuo_transformed = np.zeros((labels_feihuo.size, 15))
labels_feihuo_transformed[np.arange(labels_feihuo.size), labels_feihuo] = 1

df = pd.read_csv('datasets/questions_labeled/train_jiuan.csv')
labels_jiuan = df['label'].to_numpy()
labels_jiuan_transformed = np.zeros((labels_jiuan.size, 15))
labels_jiuan_transformed[np.arange(labels_jiuan.size), labels_jiuan] = 1

df = pd.read_csv('datasets/questions_labeled/train_sec3.csv')
labels_sec3 = df['label'].to_numpy()
labels_sec3_transformed = np.zeros((labels_sec3.size, 15))
labels_sec3_transformed[np.arange(labels_sec3.size), labels_sec3] = 1



w = np.ones(3)
w_abs_diffs = []

for i in range(10):
    prev_w = w
    labels_truth = np.argmax(w[0] * labels_feihuo_transformed + w[1] * labels_jiuan_transformed + w[2] * labels_sec3_transformed, axis=1)
    feihuo_different = num_labels - np.sum(labels_truth == labels_feihuo)
    jiuan_different = num_labels - np.sum(labels_truth == labels_jiuan)
    sec3_different = num_labels - np.sum(labels_truth == labels_sec3)
    max_diff = max(feihuo_different, jiuan_different, sec3_different)
    print("iteration ", i)
    print("feihuo diff: ", feihuo_different)
    print("jiuan diff: ", jiuan_different)
    print("sec3 diff: ", sec3_different)
    w[0] = -np.log(feihuo_different/max_diff + 1e-5)
    w[1] = -np.log(jiuan_different/max_diff + 1e-5)
    w[2] = -np.log(sec3_different/max_diff + 1e-5)
    print("current w: ", w)
    w_diff = np.subtract(prev_w, w)
    w_abs_diff = np.abs(np.sum(w_diff))
    w_abs_diffs.append(w_abs_diff)

dict_tmp = {'question': response, 'label': labels_truth}
df = pd.DataFrame(dict_tmp)
df.to_csv("datasets/questions_labeled/train_pm_human.csv")