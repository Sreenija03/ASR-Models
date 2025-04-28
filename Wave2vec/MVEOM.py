import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Function to load cosine similarity matrix from file
def load_cosine_similarity_matrix(file_path):
    try:
        cosine_similarity_matrix = np.loadtxt(file_path)
        return cosine_similarity_matrix
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to compute mean and variance of cosine similarity matrix
def compute_mean_and_variance(cosine_similarity_matrix):
    non_zero_values = cosine_similarity_matrix[cosine_similarity_matrix != 0]
    mean = np.mean(non_zero_values)
    variance = np.var(non_zero_values)
    return mean, variance

# Function to flatten and filter cosine similarity matrix to 1D array
def flatten_and_filter(matrix):
    flattened_matrix = matrix.flatten()
    filtered_matrix = flattened_matrix[flattened_matrix != 0]  # Filter out non-zero values
    return filtered_matrix

# Function to compute mean, variance, and difference in variance for each speaker
def compute_mean_variance_for_speakers(folder_path):
    data = {
        'Speaker': [],
        'Category': [],
        'Mean': [],
        'Variance': [],
        'Variance Difference': []
    }

    speaker_counter = 1
    speaker_folders = sorted([os.path.join(folder_path, speaker) for speaker in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, speaker))])

    for speaker_folder in speaker_folders:
        variances = {}
        for subfolder in ['Sent1', 'Different']:
            matrix_file = os.path.join(speaker_folder, f"{subfolder}_cosine_similarity_matrix.txt")
            cosine_similarity_matrix = load_cosine_similarity_matrix(matrix_file)

            if cosine_similarity_matrix is None:
                continue  # Skip if matrix failed to load

            mean, variance = compute_mean_and_variance(cosine_similarity_matrix)

            data['Speaker'].append(f"Speaker {speaker_counter}")
            data['Category'].append(subfolder)
            data['Mean'].append(mean)
            data['Variance'].append(variance)
            variances[subfolder] = variance

            # Append None for 'Variance Difference' in the current iteration
            data['Variance Difference'].append(None)

        # Ensure both 'Sent1' and 'Different' categories are recorded
        if 'Sent1' in variances and 'Different' in variances:
            variance_difference = (variances['Different'] - variances['Sent1'])

            # Update the 'Variance Difference' value for the first occurrence of the speaker
            speaker_indices = [index for index, value in enumerate(data['Speaker']) if value == f"Speaker {speaker_counter}"]
            if speaker_indices:
                data['Variance Difference'][speaker_indices[0]] = variance_difference

        speaker_counter += 1

    return data

# Function to save mean and variance data to Excel
def save_mean_variance_to_excel(mean_variance_data, output_excel_file):
    df = pd.DataFrame(mean_variance_data)
    df.to_excel(output_excel_file, index=False)
    print(f"Mean and Variance results exported to: {output_excel_file}")

# Function to perform equality test for means and count accept/reject decisions
def speaker_equality_test(folder_path):
    data = {
        'Speaker Comparison': [],
        'T-Statistic Different': [],
        'P-Value Different': [],
        'T-Statistic Sent1': [],
        'P-Value Sent1': [],
        'Decision Different': [],
        'Decision Sent1': []
    }

    accept_reject_counts = {
        'Different Accept H0': 0,
        'Different Reject H0': 0,
        'Sent1 Accept H0': 0,
        'Sent1 Reject H0': 0
    }

    speaker_folders = [os.path.join(folder_path, speaker) for speaker in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, speaker))]

    for i in range(len(speaker_folders)):
        for j in range(i + 1, len(speaker_folders)):
            # Define file paths
            speaker1_different_file = os.path.join(speaker_folders[i], "Different_cosine_similarity_matrix.txt")
            speaker2_different_file = os.path.join(speaker_folders[j], "Different_cosine_similarity_matrix.txt")
            speaker1_sent1_file = os.path.join(speaker_folders[i], "Sent1_cosine_similarity_matrix.txt")
            speaker2_sent1_file = os.path.join(speaker_folders[j], "Sent1_cosine_similarity_matrix.txt")

            # Load cosine similarity matrices
            cosine_sim_speaker1_diff = load_cosine_similarity_matrix(speaker1_different_file)
            cosine_sim_speaker2_diff = load_cosine_similarity_matrix(speaker2_different_file)
            cosine_sim_speaker1_sent1 = load_cosine_similarity_matrix(speaker1_sent1_file)
            cosine_sim_speaker2_sent1 = load_cosine_similarity_matrix(speaker2_sent1_file)

            if (cosine_sim_speaker1_diff is None or cosine_sim_speaker2_diff is None or
                cosine_sim_speaker1_sent1 is None or cosine_sim_speaker2_sent1 is None):
                continue  # Skip if any matrix failed to load

            # Flatten and filter cosine similarity matrices
            flat_cos_sim_speaker1_diff = flatten_and_filter(cosine_sim_speaker1_diff)
            flat_cos_sim_speaker2_diff = flatten_and_filter(cosine_sim_speaker2_diff)
            flat_cos_sim_speaker1_sent1 = flatten_and_filter(cosine_sim_speaker1_sent1)
            flat_cos_sim_speaker2_sent1 = flatten_and_filter(cosine_sim_speaker2_sent1)

            # Perform t-tests
            t_statistic_diff, p_value_diff = ttest_ind(flat_cos_sim_speaker1_diff, flat_cos_sim_speaker2_diff)
            t_statistic_sent1, p_value_sent1 = ttest_ind(flat_cos_sim_speaker1_sent1, flat_cos_sim_speaker2_sent1)

            # Round values to 5 decimal places
            t_statistic_diff = round(t_statistic_diff, 5)
            p_value_diff = round(p_value_diff, 5)
            t_statistic_sent1 = round(t_statistic_sent1, 5)
            p_value_sent1 = round(p_value_sent1, 5)

            # Store results in dictionary
            data['Speaker Comparison'].append(f"Speaker {i+1} vs Speaker {j+1}")
            data['T-Statistic Different'].append(t_statistic_diff)
            data['P-Value Different'].append(p_value_diff)
            data['T-Statistic Sent1'].append(t_statistic_sent1)
            data['P-Value Sent1'].append(p_value_sent1)

            # Determine decision based on p-values (using alpha = 0.05)
            if p_value_diff < 0.05:
                data['Decision Different'].append('Reject H0')
                accept_reject_counts['Different Reject H0'] += 1
            else:
                data['Decision Different'].append('Accept H0')
                accept_reject_counts['Different Accept H0'] += 1

            if p_value_sent1 < 0.05:
                data['Decision Sent1'].append('Reject H0')
                accept_reject_counts['Sent1 Reject H0'] += 1
            else:
                data['Decision Sent1'].append('Accept H0')
                accept_reject_counts['Sent1 Accept H0'] += 1

    return data, accept_reject_counts

# Function to save equality test results to Excel
def save_equality_test_results_to_excel(equality_test_data, accept_reject_counts, output_excel_file):
    df = pd.DataFrame(equality_test_data)
    df.to_excel(output_excel_file, index=False)
    print(f"Equality test results exported to: {output_excel_file}")

    # Print accept and reject counts
    print("Accept and Reject Counts:")
    for key, value in accept_reject_counts.items():
        print(f"{key}: {value}")

# Main function to combine all operations
def main(folder_path):
    result_folder = os.path.join(folder_path, "Results")
    os.makedirs(result_folder, exist_ok=True)

    # Compute and save mean and variance
    mean_variance_data = compute_mean_variance_for_speakers(folder_path)
    mean_variance_output_file = os.path.join(result_folder, "Mean_Variance_Results.xlsx")
    save_mean_variance_to_excel(mean_variance_data, mean_variance_output_file)

    # Compute and save equality test results
    equality_test_data, accept_reject_counts = speaker_equality_test(folder_path)
    equality_test_output_file = os.path.join(result_folder, "Equality_Test_Results.xlsx")
    save_equality_test_results_to_excel(equality_test_data, accept_reject_counts, equality_test_output_file)

# Example usage
if __name__ == "__main__":
    folder_path = "/Users/sree/Desktop/Cosine3 Layer1"  
    main(folder_path)

