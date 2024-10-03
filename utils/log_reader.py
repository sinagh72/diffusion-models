import os
import re


def sum_metric_per_folder(root_folder, keyword, metric):
    # Dictionary to store metric sums per folder
    folder_metric_sums = {}

    # Walk through the root folder to find all text files
    for filename in  os.listdir(root_folder):
        if keyword in filename and filename.endswith('.txt'):
            file_path = os.path.join(root_folder, filename)

            # Read the file content
            with open(file_path, 'r') as file:
                content = file.read()

            # Extract metric values from the content
            pattern = rf"'{metric}':\s([\d.]+)"
            metric_values = re.findall(pattern, content)
            # Sum the metric values for the current folder
            folder_metric_sums[filename] = sum(map(float, metric_values))

    # Print the sum for each folder
    for folder, metric_sum in folder_metric_sums.items():
        print(f"For folder '{folder}', the sum of '{metric}' is: {metric_sum}")


# Usage example
root_folder = '../log_100'  # Replace with the path to your root folder
keyword = 'diffusion_2'  # Replace with the keyword in the filenames
metric = 'auc'  # Change this to 'f1', 'precision', etc.

sum_metric_per_folder(root_folder, keyword, metric)
