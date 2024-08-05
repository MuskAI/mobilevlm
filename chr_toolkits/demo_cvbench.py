import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cv_bench_results.csv')

# Define a function to calculate accuracy for a given source
def calculate_accuracy(df, source):
    source_df = df[df['source'] == source]
    accuracy = source_df['result'].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
    return accuracy

# Calculate accuracy for each source
accuracy_2d_ade = calculate_accuracy(df, 'ADE20K')
accuracy_2d_coco = calculate_accuracy(df, 'COCO')
accuracy_3d_omni = calculate_accuracy(df, 'Omni3D')

# Calculate the accuracy for each type
accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
accuracy_3d = accuracy_3d_omni

# Compute the combined accuracy as specified
combined_accuracy = (accuracy_2d + accuracy_3d) / 2

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
