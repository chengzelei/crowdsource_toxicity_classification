import pandas as pd

raw_file = '/home/jys3649/projects/chatguard/datasets/templates/en_templates_mutate_raw_file.csv'
df = pd.read_csv(raw_file)

#delet rows with 'invalid' as true
df = df[df['invalid'] != True]
