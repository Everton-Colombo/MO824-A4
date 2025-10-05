import os
import re
from statistics import mean

# Directory containing the logs

def find_last_cost_reverse(file_path):
    pattern = r"cost=\[(\d+\.\d+)\]"  # Regular expression to match cost=[<float>]

    with open(file_path, 'r') as file:
        for line in reversed(list(file.readlines())):  # Read lines in reverse order
            match = re.search(pattern, line)
            if match:
                return f"{match.group(1)}"  # Return the first match (last occurrence in the file)

    return None  # Return None if no match is found

# Data structure: {instance_size: [(cost, time), ...]}
results = {}
instances = set()
LOGS_DIR = 'logs'

for folder in os.listdir(LOGS_DIR):
    log_dir = os.path.join(LOGS_DIR, folder)
    folder = folder.replace('_logs', '')
    results[folder] = []
    for filename in os.listdir(log_dir):
        instances.add(filename.replace('.log', ''))
        if not filename.endswith('.log'):
            continue
        # Extract last cost from the log file
        cost = find_last_cost_reverse(os.path.join(log_dir, filename))
        results[folder].append(cost) 

       
# Print summary
instances = sorted(instances)
keys = ['std', 'std2', 'latin_hypercube', 'steady_state', 'adaptive', 'adaptive_steady', 'latin_hypercube_steady']
lines = []
for i, instance in enumerate(instances):
    # Get the values for the current instance
    line = f"{instance} & & "

    for folder in keys:
        
        line += f"{results[folder][i]} & "
    line = line[:-2] + " \\\\"
    lines.append(line)

for line in lines:
    max_cost, i = max((float(cost), i) for i, cost in enumerate(line.split('&')[2:]))  # Skip the first two columns
    min_cost, j = min((float(cost), i) for i, cost in enumerate(line.split('&')[2:]))  # Skip the first two columns
    line[i] = line[i].replace(f"{max_cost}", f"\\textcolor{{blue}}{{{max_cost}}}")
    line[j] = line[j].replace(f"{min_cost}", f"\\textcolor{{red}}{{{min_cost}}}")
    print(' & '.join(line))
print("\n")
