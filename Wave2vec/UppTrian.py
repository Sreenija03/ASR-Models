import os
import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity_matrix(a, b):
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    out = np.matmul(a, b.transpose())
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    scale = np.matmul(norm_a, norm_b.transpose())
    out = out / scale
    return out

def load_summed_outputs(folder_path):
    summed_outputs = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            summed_output = np.load(file_path)
            summed_output = summed_output.reshape(-1)  # Flatten the array to shape (512,)
            summed_outputs.append(summed_output)
    return np.array(summed_outputs)

def compute_upper_triangular_cosine_similarity_matrix(summed_outputs):
    cosine_similarities_matrix = cosine_similarity_matrix(summed_outputs, summed_outputs)
    n = cosine_similarities_matrix.shape[0]
    upper_triangular_matrix = np.triu(cosine_similarities_matrix, k=1)
    np.fill_diagonal(upper_triangular_matrix, 0)
    return upper_triangular_matrix

def compute_histograms_from_cosine_matrix(cosine_similarities_matrix, output_folder, category):
    os.makedirs(output_folder, exist_ok=True)
    n, m = cosine_similarities_matrix.shape
    upper_indices = np.triu_indices(n=n, k=1, m=m)

    matrix_values = cosine_similarities_matrix[upper_indices]
    bins = np.linspace(0, 1, num=1000)

    # Compute the histogram
    histogram, bins = np.histogram(matrix_values, bins='auto')
    histogram = histogram / matrix_values.size  # Normalize histogram

    # Compute the cumulative frequency
    cumulative_frequency = np.cumsum(histogram)
    cumulative_frequency = cumulative_frequency / cumulative_frequency[-1]  # Normalize to 1

    # Plot and save the histogram and cumulative frequency line
    plt.figure()
    plt.hist(bins[:-1], bins, weights=histogram, alpha=0.6, label='Histogram')
    plt.plot(bins[:-1] + (bins[1] - bins[0]) / 2, cumulative_frequency, color='r', label='Cumulative Frequency')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram and Cumulative Frequency for {category}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_file_path = os.path.join(output_folder, f"{category}_histogram_cumulative.jpg")
    plt.savefig(output_file_path)
    plt.close()
    print(f"Histogram and Cumulative Frequency for {category} saved to: {output_file_path}")

def compute_cosine_similarity_matrix(root_folder, output_folder):
    for speaker_folder in sorted(os.listdir(root_folder)):
        speaker_path = os.path.join(root_folder, speaker_folder)
        if os.path.isdir(speaker_path):
            print(f"Processing speaker: {speaker_folder}")
            speaker_output_folder = os.path.join(output_folder, speaker_folder)
            os.makedirs(speaker_output_folder, exist_ok=True)
            for subfolder in ['Sent1', 'Different']:
                subfolder_path = os.path.join(speaker_path, subfolder)
                if os.path.isdir(subfolder_path):
                    summed_outputs = load_summed_outputs(subfolder_path)
                    upper_triangular_matrix = compute_upper_triangular_cosine_similarity_matrix(summed_outputs)
                    output_file_path = os.path.join(speaker_output_folder, f"{subfolder}_cosine_similarity_matrix.txt")
                    np.savetxt(output_file_path, upper_triangular_matrix, fmt='%.6f')
                    print(f"Upper Triangular Cosine Similarities Matrix for {subfolder} saved to: {output_file_path}")

                    # Compute and save histograms
                    histogram_folder = os.path.join(speaker_output_folder, f"{subfolder}_histograms")
                    compute_histograms_from_cosine_matrix(upper_triangular_matrix, histogram_folder, subfolder)

# Example usage
root_folder = "/Users/sree/Desktop/Results2Layer1"
output_folder = "/Users/sree/Desktop/Cosine3Uppertriangular"
compute_cosine_similarity_matrix(root_folder, output_folder)
