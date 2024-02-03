import re
import pandas as pd

# Open the file in read mode
name = "c_mtt"
with open(f'{name}.log', 'r') as file:
    lines = file.readlines()

# Initialize an empty list to store the data
data = []

# Regular expression patterns
pattern_processing = r"Processing (c\d+\.txt) with (\d+) customers and (\d+) vehicles"
pattern_travel_time = r"Total travel time of all routes: (\d+\.\d+)min"
pattern_solution_status = r"Solution status: (\d+)"

# Initialize processing_found flag and temporary variables
processing_found = False
filename = n_customers = n_vehicles = solution_status = None

# Iterate over each line in the file
for line in lines:
    if not line.startswith('Processing') and not line.startswith('Solution status') and not line.startswith('Total travel time'):
        continue
    if not processing_found:
        match = re.search(pattern_processing, line)
        if match:
            filename, n_customers, n_vehicles = match.groups()
            processing_found = True
    else:
        match = re.search(pattern_solution_status, line)
        if match:
            solution_status = int(match.group(1))
            if solution_status != 1:
                travel_time = None
                data.append([filename, n_customers, n_vehicles, travel_time])
                processing_found = False
        else:
            if solution_status == 1:
                match = re.search(pattern_travel_time, line)
                if match:
                    travel_time = match.group(1)
                data.append([filename, n_customers, n_vehicles, travel_time])
                processing_found = False

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data, columns=['filename', 'n_customers', 'n_vehicles', 'travel_time'])

# Convert columns to appropriate data types
df['n_customers'] = df['n_customers'].astype(int)
df['n_vehicles'] = df['n_vehicles'].astype(int)
df['travel_time'] = pd.to_numeric(df['travel_time'], errors='coerce')

# Drop duplicate rows with the same triplet (filename, n_customers, n_vehicles)
df = df.drop_duplicates(subset=['filename', 'n_customers', 'n_vehicles'])

# Sort the DataFrame by the triplet (filename, n_customers, n_vehicles)
df = df.sort_values(by=['filename', 'n_customers', 'n_vehicles'])

print(df)
# Save the DataFrame to a CSV file
df.to_csv(f'{name}.csv', index=False)
