import pandas as pd
import csv


# read the xlsx file
df = pd.read_excel('/home/jys3649/projects/chatguard/datasets/templates/en_templates.xlsx')['Template'].to_list()
# write the csv file
with open('/home/jys3649/projects/chatguard/datasets/templates/en_templates.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "template"])
    for id, template in enumerate(df):
        writer.writerow([id, template])

