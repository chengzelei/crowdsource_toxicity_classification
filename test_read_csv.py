import pandas as pd

df = pd.read_csv('/home/zck7060/crowdsource_toxicity_classification/datasets/questions_labeled/train_multi_all.csv')
# Specify the columns to keep
columns_to_keep = ['question', 'label_4', 'label_5', 'label_6']

# Select only the desired columns
filtered_df = df[columns_to_keep]

# Save the filtered dataframe back to CSV if needed
filtered_df.to_csv('/home/zck7060/crowdsource_toxicity_classification/datasets/questions_labeled/train_multi_llm.csv', index=False)