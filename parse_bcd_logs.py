import re
import pandas as pd
import argparse
import re
import pandas as pd


parser = argparse.ArgumentParser(description='Parse BCD logs')
parser.add_argument('logfile', type=str, help='Path to the log file')
args = parser.parse_args()

logfile = args.logfile

with open(logfile, 'r') as file:
    lines = file.readlines()

# Define the column names for the DataFrame
columns = ["k", "t", "c'x", "c'x (H)", "lobj", "|Ax - b|", "|cx-C|", "error", "rhol", "rhom", "tau", "iter"]

# Initialize an empty DataFrame
df = pd.DataFrame(columns=columns)

# Regular expression to match the lines containing the data
regex = re.compile(r'^(\d{3})\s+([\+\-]?\d+\.\d+|[\+\-]inf)\s+([\+\-]?\d+\.\d+|[\+\-]inf)\s+([\+\-]?\d+\.\d+|[\+\-]inf)\s+([\+\-]?\d+\.\d+|[\+\-]inf)\s+([\+\-]?\d+\.\d+e[\+\-]\d+|[\+\-]inf)\s+([\+\-]?\d+\.\d+e[\+\-]\d+|[\+\-]inf)\s+([\+\-]?\d+\.\d+e[\+\-]\d+|[\+\-]inf)\s+([\+\-]?\d+\.\d+e[\+\-]\d+|[\+\-]inf)\s+([\+\-]?\d+\.\d+e[\+\-]\d+|[\+\-]inf)\s+([\+\-]?\d+\.\d+e[\+\-]\d+|[\+\-]inf)\s+(\d{4})$')

for line in lines:
    match = regex.match(line)
    if match:
        # Extract the data from the matched line
        data = match.groups()
        # Convert the dictionary to a DataFrame and append it
        df_to_append = pd.DataFrame([dict(zip(columns, data))])
        df = pd.concat([df, df_to_append], ignore_index=True)

print(df)
df.to_csv(logfile.replace(".txt", ".csv"), index=False)

# FIXME: IMPORTANT! remember to leave the last 5 columns of logs!!!
