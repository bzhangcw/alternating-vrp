import pandas as pd
import re
import argparse

def parse_gurobi_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Define the column names for the DataFrame
    columns = ["Incumbent", "BestBd", "Gap", "It/Node", "Time"]

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=columns)

    # Regular expression to match the lines containing the data
    regex = re.compile(r'\s*(-|\d+\.?\d+)\s+(-|\d+\.?\d+)\s+(-|\d+\.?\d+)\%?\s+(-|\d+\.?\d+)\s+(\d+)s')
    for line in lines:
        match = regex.match(line)
        if match:
            # Extract the data from the matched line
            data = match.groups()
            # Convert the dictionary to a DataFrame and append it
            df_to_append = pd.DataFrame([dict(zip(columns, data))])
            df = pd.concat([df, df_to_append], ignore_index=True)

    return df


# Create an argument parser
parser = argparse.ArgumentParser(description='Parse Gurobi log file')

# Add an argument for the log file path
parser.add_argument('log_file', type=str, help='Path to the Gurobi log file')

# Parse the command line arguments
args = parser.parse_args()

log_file = args.log_file

# Call the parse_gurobi_log function with the provided log file path
df = parse_gurobi_log(log_file)

# Save the DataFrame to a CSV file with the same name as the log file, but with a .csv extension
csv_file = log_file.replace(".log", ".csv")
df.to_csv(csv_file, index=False)
# Usage
df = parse_gurobi_log(log_file)
df.to_csv(log_file.replace(".log", ".csv"), index=False)
