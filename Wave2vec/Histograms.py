#Code for calculating the histograms of Sent1 and Different files of all the speakers

import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import matplotlib.pyplot as plt

class SimpleWav2Vec2CNN(nn.Module):
    def __init__(self):
        super(SimpleWav2Vec2CNN, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def forward(self, input_values):
        outputs = self.wav2vec2.feature_extractor(input_values)
        return outputs

def model_out(path_single_audio, sample_rate):
    vector_audio, sr = librosa.load(path_single_audio, sr=sample_rate)
    vector_audio = np.expand_dims(vector_audio, axis=0)
    vector_tensor = torch.tensor(vector_audio, dtype=torch.float32)
    outputs_tensor = model(vector_tensor)
    summed_outputs = torch.sum(outputs_tensor, dim=2)
    summed_outputs = summed_outputs.detach().numpy()
    return summed_outputs

def list_all_wav_files(root_folder):
    wav_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.wav'):
                wav_files.append(os.path.join(dirpath, filename))
    return wav_files

def save_summed_outputs_as_npy(files, sample_rate, output_root_folder):
    for file in files:
        summed_outputs = model_out(file, sample_rate)
        relative_path = os.path.relpath(file, folder_path)
        new_file_name = os.path.splitext(relative_path)[0] + '_results.npy'
        npy_path = os.path.join(output_root_folder, new_file_name)
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, summed_outputs)
        print(f'Saved: {npy_path}')

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

def save_matrix(matrix, output_folder, label):
    np.savetxt(os.path.join(output_folder, f'cosine_similarity_matrix_{label}.txt'), matrix, fmt='%.6f')

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

def compute_cosine_similarity_matrix_combined(root_folder, output_folder):
    combined_summed_outputs_sent1 = []
    combined_summed_outputs_different = []

    for speaker_folder in sorted(os.listdir(root_folder)):
        speaker_path = os.path.join(root_folder, speaker_folder)
        if os.path.isdir(speaker_path):
            print(f"Processing speaker: {speaker_folder}")
            for subfolder in ['Sent1', 'Different']:
                subfolder_path = os.path.join(speaker_path, subfolder)
                if os.path.isdir(subfolder_path):
                    summed_outputs = load_summed_outputs(subfolder_path)
                    if subfolder == 'Sent1':
                        combined_summed_outputs_sent1.append(summed_outputs)
                    else:
                        combined_summed_outputs_different.append(summed_outputs)
    
    # Combine all summed outputs for each category
    if combined_summed_outputs_sent1:
        combined_summed_outputs_sent1 = np.concatenate(combined_summed_outputs_sent1, axis=0)
        cosine_similarities_matrix_sent1 = cosine_similarity_matrix(combined_summed_outputs_sent1, combined_summed_outputs_sent1)
        save_matrix(cosine_similarities_matrix_sent1, output_folder, 'Sent1')
        compute_histograms_from_cosine_matrix(cosine_similarities_matrix_sent1, output_folder, 'Sent1')
    
    if combined_summed_outputs_different:
        combined_summed_outputs_different = np.concatenate(combined_summed_outputs_different, axis=0)
        cosine_similarities_matrix_different = cosine_similarity_matrix(combined_summed_outputs_different, combined_summed_outputs_different)
        save_matrix(cosine_similarities_matrix_different, output_folder, 'Different')
        compute_histograms_from_cosine_matrix(cosine_similarities_matrix_different, output_folder, 'Different')

model = SimpleWav2Vec2CNN()
folder_path = "/Users/sree/Desktop/Audio"
output_folder = "/Users/sree/Desktop/Results"
out_folder = "/Users/sree/Desktop/Cosine5"
wav_files = list_all_wav_files(folder_path)
# save_summed_outputs_as_npy(wav_files, 16000, output_folder)
compute_cosine_similarity_matrix_combined(output_folder, out_folder)