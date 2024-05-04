import pandas as pd

def extract_columns_from_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Extract the 'response' and 'label' columns
    extracted_df = df[['response', 'label']]
    
    # Save the extracted columns to a new CSV file
    extracted_df.to_csv(output_file, index=False)

# Paths to your CSV files
input_file1 = 'all_labeled_gpt-3.5-turbo.csv'
input_file2 = 'init_all_vicuna-7b-v1.3_labeled.20230920.csv'

# Paths to save the new CSV files
output_file1 = 'evaluate.csv'
output_file2 = 'train.csv'

# Extract columns and save to new files
extract_columns_from_csv(input_file1, output_file1)
extract_columns_from_csv(input_file2, output_file2)

print("Extraction complete!")
