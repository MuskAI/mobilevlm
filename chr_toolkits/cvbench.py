import pandas as pd
import json
import re
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Process JSON file path.')
    parser.add_argument('--json_path', type=str, help='Path to the JSONL file')
    parser.add_argument('--result_path', type=str, help='Path to the result file')
    return parser.parse_args()

# Load the JSON file into a DataFrame
def load_data(json_path):
    data = []
    with open(json_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return pd.DataFrame(data)

def filter_text(df):
    df["filtered_text"] = df["text"].apply(lambda x: " ".join(re.findall(r'[a-zA-Z]', x)))
    return df

# Define a function to calculate accuracy for a given source
def calculate_accuracy(df, source):
    source_df = df[df['source'] == source]
    count = 0
    for i in range(len(source_df)):
        if list(source_df['filtered_text'])[i] in list(source_df['answer'])[i]:
            count += 1
    accuracy = count / len(source_df)  # Assuming 'result' is 1 for correct and 0 for incorrect
    return accuracy

def main():
    args = parse_args()
    df = load_data(args.json_path)
    df = filter_text(df)
    
    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(df, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(df, 'COCO')
    accuracy_3d_omni = calculate_accuracy(df, 'Omni3D')

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    # Compute the combined accuracy as specified
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2

    # Open the result file and redirect stdout to this file
    with open(args.result_path, 'w') as f:
        sys.stdout = f  # Redirect stdout to the file
        # Print the results
        print(f"CV-Bench Accuracy: {combined_accuracy:.4f}")
        print()
        print(f"Type Accuracies:")
        print(f"2D Accuracy: {accuracy_2d:.4f}")
        print(f"3D Accuracy: {accuracy_3d:.4f}")
        print()
        print(f"Source Accuracies:")
        print(f"ADE20K Accuracy: {accuracy_2d_ade:.4f}")
        print(f"COCO Accuracy: {accuracy_2d_coco:.4f}")
        print(f"Omni3D Accuracy: {accuracy_3d_omni:.4f}")
        sys.stdout = sys.__stdout__  # Reset stdout to default

if __name__ == '__main__':
    main()