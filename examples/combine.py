import os
import pandas as pd
import re
# Define the directory containing the CSV files
directory = 'results/sub'  # Replace with your actual path

# Function to extract sort key if needed (custom sort logic, can modify if necessary)
def extract_sort_key(filename):
    # Extract the numerical part of the filename (assumes format '4qs_Test_i<number>_...')
    
    match = re.search(r'i(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')  # Sort non-matching filenames to the end

# Get a sorted list of all files in the directory using the extract_sort_key
sorted_files = sorted(os.listdir(directory), key=extract_sort_key)

# List of effects to loop through
# effect_range = ['none', 'weak']

# Loop through each effect in the range
# effect = 'weak'
effect_range = ['none', 'weak'] # , 'strong'

# Loop through each effect in the range
for effect in effect_range:
    # Reset the DataFrame list for each effect
    dfs = []
    # i=1
    # Loop through the sorted files and process matching CSV files
    for filename in sorted_files:
        if filename.endswith(".csv") and filename.startswith("t_Test_i") and f'_{effect}_combined_results_01.csv' in filename:  
            print(f"Processing file: {filename}")  # Debugging output
            # print(i)
            # i+=1
            # Read the CSV file into a DataFrame and append it to the list
            df = pd.read_csv(os.path.join(directory, filename))
            dfs.append(df)

    # Check if there are any DataFrames to concatenate
    
    if dfs:
        # Concatenate all DataFrames in the list along the rows
        combined_df = pd.concat(dfs, ignore_index=True)
        # if effect == 'none':
        #     combined_df.loc[combined_df['diff'] == 0.000001, 'diff'] = 0.00001
        #     print(combined_df)
        # Output file path for the combined DataFrame
        output_file = f'results/t_Test_iall_{effect}_combined_results.csv'

        # Write the combined DataFrame to a CSV file
        combined_df.to_csv(output_file, index=False)

        print(f'Combined results saved to {output_file}')
    else:
        print(f"No files found for effect '{effect}'")
