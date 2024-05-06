import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('datasets/questions_labeled/train_gpt4.csv')
response = df['question']
labels_gpt4 = df['label'].to_numpy()
num_labels = len(labels_gpt4)
labels_gpt4_transformed = np.zeros((labels_gpt4.size, 15))
labels_gpt4_transformed[np.arange(labels_gpt4.size), labels_gpt4] = 1

df = pd.read_csv('datasets/questions_labeled/train_turbo.csv')
labels_gpt4turbo = df['label'].to_numpy()
labels_gpt4turbo_transformed = np.zeros((labels_gpt4turbo.size, 15))
labels_gpt4turbo_transformed[np.arange(labels_gpt4turbo.size), labels_gpt4turbo] = 1

df = pd.read_csv('datasets/questions_labeled/train_claude.csv')
labels_claude = df['label'].to_numpy()
labels_claude_transformed = np.zeros((labels_claude.size, 15))
labels_claude_transformed[np.arange(labels_claude.size), labels_claude] = 1



w = np.ones(3)
w_abs_diffs = []

for i in range(10):
    prev_w = w
    labels_truth = np.argmax(w[0] * labels_gpt4_transformed + w[1] * labels_gpt4turbo_transformed + w[2] * labels_claude_transformed, axis=1)
    gpt4_different = num_labels - np.sum(labels_truth == labels_gpt4)
    gpt4turbo_different = num_labels - np.sum(labels_truth == labels_gpt4turbo)
    claude_different = num_labels - np.sum(labels_truth == labels_claude)
    max_diff = max(gpt4_different, gpt4turbo_different, claude_different)
    print("iteration ", i)
    print("gpt4 diff: ", gpt4_different)
    print("gpt4 turbo diff: ", gpt4turbo_different)
    print("claude diff: ", claude_different)
    w[0] = -np.log(gpt4_different/max_diff + 1e-5)
    w[1] = -np.log(gpt4turbo_different/max_diff + 1e-5)
    w[2] = -np.log(claude_different/max_diff + 1e-5)
    print("current w: ", w)
    w_diff = np.subtract(prev_w, w)
    w_abs_diff = np.abs(np.sum(w_diff))
    w_abs_diffs.append(w_abs_diff)

dict_tmp = {'question': response, 'label': labels_truth}
df = pd.DataFrame(dict_tmp)
df.to_csv("datasets/questions_labeled/train_pm_llm.csv")

