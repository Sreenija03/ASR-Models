# import os
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# from dtw import accelerated_dtw

# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         outputs = self.wav2vec2.feature_extractor(input_values)
#         return outputs

# def model_out(model, path_single_audio, sample_rate):
#     vector_audio, sr = librosa.load(path_single_audio, sr=sample_rate)
#     vector_audio = np.expand_dims(vector_audio, axis=0)
#     vector_tensor = torch.tensor(vector_audio, dtype=torch.float32)
#     outputs_tensor = model(vector_tensor)
#     print("Original tensor", outputs_tensor.shape)

#     # Compute the average across columns
#     avg_outputs = torch.mean(outputs_tensor, dim=2, keepdim=True)
#     print("Avg Outputs", avg_outputs.shape)

#     # Subtract the average from each column
#     centered_outputs = outputs_tensor - avg_outputs

#     centered_outputs = centered_outputs.detach().numpy()
#     print(f"Centered outputs dimensions for {path_single_audio}: {centered_outputs.shape}")
#     return centered_outputs

# def list_all_wav_files(root_folder):
#     wav_files = []
#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             if filename.endswith('.wav'):
#                 wav_files.append(os.path.join(dirpath, filename))
#     return wav_files

# def save_centered_outputs_as_npy(model, files, sample_rate, output_root_folder):
#     for file in files:
#         centered_outputs = model_out(model, file, sample_rate)
#         relative_path = os.path.relpath(file, folder_path)
#         new_file_name = os.path.splitext(relative_path)[0] + '_results.npy'
#         npy_path = os.path.join(output_root_folder, new_file_name)
#         os.makedirs(os.path.dirname(npy_path), exist_ok=True)
#         np.save(npy_path, centered_outputs)
#         print(f'Saved: {npy_path}')

# def dtw_cosine_distance(a, b):
#     # Ensure both sequences have the same number of columns
#     min_length = min(a.shape[2], b.shape[2])
#     a = a[:, :, :min_length]
#     b = b[:, :, :min_length]

#     a_flat = a.reshape(a.shape[1], -1)
#     b_flat = b.reshape(b.shape[1], -1)
#     dist, _, _, _ = accelerated_dtw(a_flat, b_flat, dist='cosine')
#     return dist

# def load_centered_outputs(folder_path):
#     centered_outputs = []
#     file_paths = []
#     for root, _, files in os.walk(folder_path):
#         for file_name in files:
#             if file_name.endswith('.npy'):
#                 file_path = os.path.join(root, file_name)
#                 centered_output = np.load(file_path, allow_pickle=True)
#                 centered_outputs.append(centered_output)
#                 file_paths.append(file_path)
#     return centered_outputs, file_paths

# def compute_dtw_cosine_distances(folder_path, output_file):
#     centered_outputs, file_paths = load_centered_outputs(folder_path)
#     n = len(centered_outputs)
    
#     # Print the size of the distance matrix
#     print(f"The size of the distance matrix will be: {n}x{n}")

#     distance_matrix = np.zeros((n, n))

#     for i in range(n):
#         for j in range(i + 1, n):
#             dist = dtw_cosine_distance(centered_outputs[i], centered_outputs[j])
#             distance_matrix[i, j] = distance_matrix[j, i] = dist

#     np.savetxt(output_file, distance_matrix, fmt='%.6f')
#     print(f"DTW Cosine Distance Matrix saved to: {output_file}")

# if __name__ == "__main__":
#     model = SimpleWav2Vec2CNN()

#     folder_path = "/Users/sree/Desktop/Audio1"
#     output_folder = "/Users/sree/Desktop/ResultsDTW"
#     distance_matrix_file = "/Users/sree/Desktop/dtw_cosine_distances.txt"

#     wav_files = list_all_wav_files(folder_path)
#     save_centered_outputs_as_npy(model, wav_files, 16000, output_folder)
#     compute_dtw_cosine_distances(output_folder, distance_matrix_file)



#HIstograms

# import os
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# from dtw import accelerated_dtw

# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         outputs = self.wav2vec2.feature_extractor(input_values)
#         return outputs

# def model_out(model, path_single_audio, sample_rate):
#     vector_audio, sr = librosa.load(path_single_audio, sr=sample_rate)
#     vector_audio = np.expand_dims(vector_audio, axis=0)
#     vector_tensor = torch.tensor(vector_audio, dtype=torch.float32)
#     outputs_tensor = model(vector_tensor)
#     print("Original tensor", outputs_tensor.shape)

#     # Compute the average across columns
#     avg_outputs = torch.mean(outputs_tensor, dim=2, keepdim=True)
#     print("Avg Outputs", avg_outputs.shape)

#     # Subtract the average from each column
#     centered_outputs = outputs_tensor - avg_outputs

#     centered_outputs = centered_outputs.detach().numpy()
#     print(f"Centered outputs dimensions for {path_single_audio}: {centered_outputs.shape}")
#     return centered_outputs

# def list_all_wav_files(root_folder):
#     wav_files = []
#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             if filename.endswith('.wav'):
#                 wav_files.append(os.path.join(dirpath, filename))
#     return wav_files

# def save_centered_outputs_as_npy(model, files, sample_rate, output_root_folder):
#     for file in files:
#         centered_outputs = model_out(model, file, sample_rate)
#         relative_path = os.path.relpath(file, folder_path)
#         new_file_name = os.path.splitext(relative_path)[0] + '_results.npy'
#         npy_path = os.path.join(output_root_folder, new_file_name)
#         os.makedirs(os.path.dirname(npy_path), exist_ok=True)
#         np.save(npy_path, centered_outputs)
#         print(f'Saved: {npy_path}')

# def dtw_cosine_distance(a, b):
#     # Ensure both sequences have the same number of columns
#     min_length = min(a.shape[2], b.shape[2])
#     a = a[:, :, :min_length]
#     b = b[:, :, :min_length]

#     a_flat = a.reshape(a.shape[1], -1)
#     b_flat = b.reshape(b.shape[1], -1)
#     dist, _, _, _ = accelerated_dtw(a_flat, b_flat, dist='cosine')
#     return dist

# def load_centered_outputs(folder_path):
#     centered_outputs = []
#     file_paths = []
#     for root, _, files in os.walk(folder_path):
#         for file_name in files:
#             if file_name.endswith('.npy'):
#                 file_path = os.path.join(root, file_name)
#                 centered_output = np.load(file_path, allow_pickle=True)
#                 centered_outputs.append(centered_output)
#                 file_paths.append(file_path)
#     return centered_outputs, file_paths

# def compute_dtw_cosine_distances(folder_path, output_file):
#     centered_outputs, file_paths = load_centered_outputs(folder_path)
#     n = len(centered_outputs)
    
#     # Print the size of the distance matrix
#     print(f"The size of the distance matrix will be: {n}x{n}")

#     distance_matrix = np.zeros((n, n))

#     for i in range(n):
#         for j in range(i + 1, n):
#             dist = dtw_cosine_distance(centered_outputs[i], centered_outputs[j])
#             distance_matrix[i, j] = distance_matrix[j, i] = dist

#     np.savetxt(output_file, distance_matrix, fmt='%.6f')
#     print(f"DTW Cosine Distance Matrix saved to: {output_file}")

#     # Plotting the histogram of the cosine distance matrix
#     plt.hist(distance_matrix[distance_matrix != 0].flatten(), bins=50, alpha=0.75)
#     plt.xlabel('Cosine Distance')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Cosine Distances')
#     plt.grid(True)

#     # Save the histogram plot
#     histogram_file = output_file.replace('.txt', '_histogram.png')
#     plt.savefig(histogram_file)
#     print(f"Histogram saved to: {histogram_file}")
#     plt.show()

# if __name__ == "__main__":
#     model = SimpleWav2Vec2CNN()

#     folder_path = "/Users/sree/Desktop/Audio"
#     output_folder = "/Users/sree/Desktop/ResultsDTWAudio"
#     distance_matrix_file = "/Users/sree/Desktop/dtw_cosine_distancesAudio.txt"

#     wav_files = list_all_wav_files(folder_path)
#     save_centered_outputs_as_npy(model, wav_files, 16000, output_folder)
#     compute_dtw_cosine_distances(output_folder, distance_matrix_file)



import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from dtwalign import dtw
from scipy.spatial.distance import cosine

class SimpleWav2Vec2CNN(nn.Module):
    def __init__(self):
        super(SimpleWav2Vec2CNN, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def forward(self, input_values):
        outputs = self.wav2vec2.feature_extractor(input_values)
        return outputs

def model_out(model, path_single_audio, sample_rate):
    vector_audio, sr = librosa.load(path_single_audio, sr=sample_rate)
    vector_audio = np.expand_dims(vector_audio, axis=0)
    vector_tensor = torch.tensor(vector_audio, dtype=torch.float32)
    outputs_tensor = model(vector_tensor)
    print("Original tensor", outputs_tensor.shape)

    # Compute the average across columns
    avg_outputs = torch.mean(outputs_tensor, dim=2, keepdim=True)
    print("Avg Outputs", avg_outputs.shape)

    # Subtract the average from each column
    centered_outputs = outputs_tensor - avg_outputs

    # Squeeze the tensor to remove dimensions of size 1
    centered_outputs = centered_outputs.squeeze(0)  # Shape will be (512, n)

    centered_outputs = centered_outputs.detach().numpy()
    print(f"Centered outputs dimensions for {path_single_audio}: {centered_outputs.shape}")
    return centered_outputs

def list_all_wav_files(root_folder):
    wav_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.wav'):
                wav_files.append(os.path.join(dirpath, filename))
    return wav_files

def save_centered_outputs_as_npy(model, files, sample_rate, output_root_folder):
    for file in files:
        centered_outputs = model_out(model, file, sample_rate)
        relative_path = os.path.relpath(file, folder_path)
        new_file_name = os.path.splitext(relative_path)[0] + '_results.npy'
        npy_path = os.path.join(output_root_folder, new_file_name)
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        
        # Save centered outputs as numpy array
        np.save(npy_path, centered_outputs)
        print(f'Saved: {npy_path}')

def cosine_distance(x, y):
    return cosine(x, y)/2

def dtw_cosine_distance(a, b):
    dist = dtw(a.T, b.T, dist=cosine_distance)
    return 1-dist.normalized_distance

def load_centered_outputs(folder_path):
    centered_outputs = []
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.npy'):
                file_path = os.path.join(root, file_name)
                centered_output = np.load(file_path, allow_pickle=True)
                centered_outputs.append(centered_output)
                file_paths.append(file_path)
    return centered_outputs, file_paths

def compute_dtw_cosine_distances(folder_path, output_file):
    centered_outputs, file_paths = load_centered_outputs(folder_path)
    n = len(centered_outputs)
    
    # Print the size of the distance matrix
    print(f"The size of the distance matrix will be: {n}x{n}")

    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = dtw_cosine_distance(centered_outputs[i], centered_outputs[j])
            distance_matrix[i, j] = distance_matrix[j, i] = dist

    np.savetxt(output_file, distance_matrix, fmt='%.6f')
    print(f"DTW Cosine Distance Matrix saved to: {output_file}")

def generate_histogram_from_matrix(matrix_file, histogram_file, x_limit=None, y_limit=None):
    distance_matrix = np.loadtxt(matrix_file)
    
    # Plotting the histogram of the cosine distance matrix
    plt.hist(distance_matrix[distance_matrix != 0].flatten(), bins=50, alpha=0.75)
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of Cosine Distances')
    plt.grid(True)
    
    if x_limit:
        plt.xlim(x_limit)
    if y_limit:
        plt.ylim(y_limit)

    plt.savefig(histogram_file)
    print(f"Histogram saved to: {histogram_file}")
    plt.show()

if __name__ == "__main__":
    model = SimpleWav2Vec2CNN()

    folder_path = "/Users/sree/Desktop/Audio"
    output_folder_sent1 = "/Users/sree/Desktop/ResultsDTWSent1"
    output_folder_different = "/Users/sree/Desktop/ResultsDTWDifferent"
    distance_matrix_file_sent1 = "/Users/sree/Desktop/dtw_cosine_distances_Sent1.txt"
    distance_matrix_file_different = "/Users/sree/Desktop/dtw_cosine_distances_Different.txt"
    histogram_file_sent1 = "/Users/sree/Desktop/dtw_cosine_distances_Sent1_histogram.png"
    histogram_file_different = "/Users/sree/Desktop/dtw_cosine_distances_Different_histogram.png"

    sent1_files = []
    different_files = []

    # Separate the files into Sent1 and Different categories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                if 'Sent1' in root:
                    sent1_files.append(os.path.join(root, file))
                elif 'Different' in root:
                    different_files.append(os.path.join(root, file))

    # Save centered outputs as numpy arrays
    save_centered_outputs_as_npy(model, sent1_files, 16000, output_folder_sent1)
    save_centered_outputs_as_npy(model, different_files, 16000, output_folder_different)

    # Compute DTW cosine distances
    compute_dtw_cosine_distances(output_folder_sent1, distance_matrix_file_sent1)
    compute_dtw_cosine_distances(output_folder_different, distance_matrix_file_different)

    # Compute the limits for the histograms
    all_distances = []
    for matrix_file in [distance_matrix_file_sent1, distance_matrix_file_different]:
        distance_matrix = np.loadtxt(matrix_file)
        all_distances.extend(distance_matrix[distance_matrix != 0].flatten())
    all_distances = np.array(all_distances)
    x_limit = [400, np.max(all_distances)]
    y_limit = [0, np.histogram(all_distances, bins=50)[0].max()]

    generate_histogram_from_matrix(distance_matrix_file_sent1, histogram_file_sent1, x_limit, y_limit)
    generate_histogram_from_matrix(distance_matrix_file_different, histogram_file_different, x_limit, y_limit)



