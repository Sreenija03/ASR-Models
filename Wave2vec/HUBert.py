# import os
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from transformers import HubertModel
# import matplotlib.pyplot as plt
# import seaborn as sns

# class SimpleHubertCNN(nn.Module):
#     def __init__(self):
#         super(SimpleHubertCNN, self).__init__()
#         self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

#     def forward(self, input_values):
#         outputs = self.hubert.feature_extractor(input_values)
#         return outputs

# def model_out(path_single_audio, sample_rate):
#     vector_audio, sr = librosa.load(path_single_audio, sr=sample_rate)
#     vector_audio = np.expand_dims(vector_audio, axis=0)
#     vector_tensor = torch.tensor(vector_audio, dtype=torch.float32)
#     outputs_tensor = model(vector_tensor)
#     summed_outputs = torch.sum(outputs_tensor, dim=2)
#     summed_outputs = summed_outputs.detach().numpy()
#     return summed_outputs

# def list_all_wav_files(root_folder):
#     wav_files = []
#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             if filename.endswith('.wav'):
#                 wav_files.append(os.path.join(dirpath, filename))
#     return wav_files

# def save_summed_outputs_as_npy(files, sample_rate, output_root_folder):
#     for file in files:
#         summed_outputs = model_out(file, sample_rate)
#         relative_path = os.path.relpath(file, folder_path)
#         new_file_name = os.path.splitext(relative_path)[0] + '_results.npy'
#         npy_path = os.path.join(output_root_folder, new_file_name)
#         os.makedirs(os.path.dirname(npy_path), exist_ok=True)
#         np.save(npy_path, summed_outputs)
#         print(f'Saved: {npy_path}')

# def cosine_similarity_matrix(a, b):
#     a = np.atleast_2d(a)
#     b = np.atleast_2d(b)
#     out = np.matmul(a, b.transpose())
#     norm_a = np.linalg.norm(a, axis=1, keepdims=True)
#     norm_b = np.linalg.norm(b, axis=1, keepdims=True)
#     scale = np.matmul(norm_a, norm_b.transpose())
#     out = out / scale
#     return out

# def load_summed_outputs(folder_path):
#     summed_outputs = []
#     for file_name in sorted(os.listdir(folder_path)):
#         if file_name.endswith('.npy'):
#             file_path = os.path.join(folder_path, file_name)
#             summed_output = np.load(file_path)
#             summed_output = summed_output.reshape(-1)  # Flatten the array to shape (512,)
#             summed_outputs.append(summed_output)
#     return np.array(summed_outputs)

# def plot_cosine_similarity_matrix(matrix, output_path, title):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the output directory exists
#     plt.figure(figsize=(20, 20))
#     sns.heatmap(matrix, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", annot_kws={"size": 10, "weight": "bold"}, linewidths=0.5, linecolor='black', square=True)
#     plt.title(title)
#     plt.xticks(rotation=90)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#     print(f"Cosine similarity matrix saved to: {output_path}")

# def compute_cosine_similarity_matrix(root_folder, output_folder):
#     all_sent1_outputs = []
#     all_different_outputs = []

#     for speaker_folder in sorted(os.listdir(root_folder)):
#         speaker_path = os.path.join(root_folder, speaker_folder)
#         if os.path.isdir(speaker_path):
#             print(f"Processing speaker: {speaker_folder}")
#             for subfolder in ['Sent1', 'Different']:
#                 subfolder_path = os.path.join(speaker_path, subfolder)
#                 if os.path.isdir(subfolder_path):
#                     summed_outputs = load_summed_outputs(subfolder_path)
#                     if subfolder == 'Sent1':
#                         all_sent1_outputs.append(summed_outputs)
#                     elif subfolder == 'Different':
#                         all_different_outputs.append(summed_outputs)

#     if all_sent1_outputs:
#         all_sent1_outputs = np.vstack(all_sent1_outputs)
#         sent1_cosine_matrix = cosine_similarity_matrix(all_sent1_outputs, all_sent1_outputs)
#         sent1_cosine_matrix[np.tril_indices_from(sent1_cosine_matrix, k=0)] = 0
#         plot_cosine_similarity_matrix(sent1_cosine_matrix, os.path.join(output_folder, "Sent1_cosine_matrix.png"), "Sent1 Cosine Similarity Matrix")

#     if all_different_outputs:
#         all_different_outputs = np.vstack(all_different_outputs)
#         different_cosine_matrix = cosine_similarity_matrix(all_different_outputs, all_different_outputs)
#         different_cosine_matrix[np.tril_indices_from(different_cosine_matrix, k=0)] = 0
#         plot_cosine_similarity_matrix(different_cosine_matrix, os.path.join(output_folder, "Different_cosine_matrix.png"), "Different Cosine Similarity Matrix")

# model = SimpleHubertCNN()
# folder_path = "/Users/sree/Desktop/Audio"
# output_folder = "/Users/sree/Desktop/HUBERT Results"
# out_folder = "/Users/sree/Desktop/COSINEHUBERT Results"
# wav_files = list_all_wav_files(folder_path)
# save_summed_outputs_as_npy(wav_files, 16000, output_folder)
# compute_cosine_similarity_matrix(output_folder, out_folder)


# import os
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from transformers import HubertModel
# import matplotlib.pyplot as plt

# class SimpleHubertCNN(nn.Module):
#     def __init__(self):
#         super(SimpleHubertCNN, self).__init__()
#         self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    
#     def forward(self, input_values):
#         outputs = self.hubert(input_values, output_hidden_states=True)
#         first_layer_output = outputs.hidden_states[1]  # Index 1 corresponds to the first hidden layer
#         return first_layer_output

# def model_out(path_single_audio, sample_rate):
#     vector_audio, sr = librosa.load(path_single_audio, sr=sample_rate)
#     vector_audio = np.expand_dims(vector_audio, axis=0)
#     vector_tensor = torch.tensor(vector_audio, dtype=torch.float32)
#     outputs_tensor = model(vector_tensor)
#     summed_outputs = torch.mean(outputs_tensor, dim=1)  # Summarize across the time dimension
#     summed_outputs = summed_outputs.detach().numpy()
#     return summed_outputs

# def list_all_wav_files(root_folder):
#     wav_files = []
#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             if filename.endswith('.wav'):
#                 wav_files.append(os.path.join(dirpath, filename))
#     return wav_files

# def save_summed_outputs_as_npy(files, sample_rate, output_root_folder):
#     for file in files:
#         reduced_outputs = model_out(file, sample_rate)
#         relative_path = os.path.relpath(file, folder_path)
#         new_file_name = os.path.splitext(relative_path)[0] + '_results.npy'
#         npy_path = os.path.join(output_root_folder, new_file_name)
#         os.makedirs(os.path.dirname(npy_path), exist_ok=True)
#         np.save(npy_path, reduced_outputs)
#         print(f'Saved: {npy_path}')

# if __name__ == "__main__":
#     model = SimpleHubertCNN()
#     folder_path = "/Users/sree/Desktop/Audio1"
#     output_folder = "/Users/sree/Desktop/ResultsHUBERTFirstLayerCS(512X1 vectors)"
#     wav_files = list_all_wav_files(folder_path)
#     save_summed_outputs_as_npy(wav_files, 16000, output_folder)





import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def list_all_npy_files(root_folder):
    npy_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.npy'):
                npy_files.append(os.path.join(dirpath, filename))
    return npy_files

def load_npy(file):
    array = np.load(file)
    return array.reshape(1, -1)

def apply_pca_and_get_2d_vectors(arrays):
    pca = PCA(n_components=2)
    reduced_array = pca.fit_transform(arrays)
    return reduced_array

def plot_scatter(data, labels, output_file, title, num_audio_files):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', num_audio_files)
    
    for i in range(num_audio_files):
        indices = np.where(labels == i + 1)
        plt.scatter(data[indices, 0], data[indices, 1], c=colors(i), alpha=0.7, label=f'Audio {i+1}')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='best', title='Audio Files')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Scatter plot saved to: {output_file}")

def process_and_plot(root_folder, output_plot_folder):
    os.makedirs(output_plot_folder, exist_ok=True)
    speakers = sorted(os.listdir(root_folder))
    
    all_data = []
    labels = []
    max_audio_files = 0
    
    for speaker in speakers:
        speaker_path = os.path.join(root_folder, speaker)
        
        if os.path.isdir(speaker_path):
            npy_files = list_all_npy_files(speaker_path)
            if len(npy_files) > max_audio_files:
                max_audio_files = len(npy_files)
            
            for i, npy_file in enumerate(npy_files):
                array = load_npy(npy_file)
                all_data.append(array)
                labels.append(i + 1)  # Assign label to each audio file index
    
    if all_data:
        all_data = np.vstack(all_data)
        labels = np.array(labels)
        print(f"Shape before PCA: {all_data.shape}")
        
        reduced_data = apply_pca_and_get_2d_vectors(all_data)
        print(f"Shape after PCA: {reduced_data.shape}")
        
        output_file = os.path.join(output_plot_folder, "combined_scatter_plot.png")
        plot_scatter(reduced_data, labels, output_file, "Combined Scatter Plot for Audio Files", max_audio_files)
    else:
        print("No data to plot.")

# Example usage:
npy_folder_path = "/Users/sree/Desktop/ResultsFirstLayerCS(512X1 vectors)"
output_plot_folder = "/Users/sree/Desktop/ScatterPlotsFirstLayerHUBERT"

process_and_plot(npy_folder_path, output_plot_folder)

# import os
# import numpy as np
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# def list_all_npy_files(root_folder):
#     npy_files = []
#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             if filename.endswith('.npy'):
#                 npy_files.append(os.path.join(dirpath, filename))
#     return npy_files

# def load_npy(file):
#     array = np.load(file)
#     return array.reshape(1, -1)

# def apply_pca_and_get_2d_vectors(arrays):
#     pca = PCA(n_components=2)
#     reduced_array = pca.fit_transform(arrays)
#     return reduced_array

# def plot_scatter(data, labels, output_file, title, speakers):
#     plt.figure(figsize=(10, 8))
#     unique_labels = np.unique(labels)
#     colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
#     for label in unique_labels:
#         indices = np.where(labels == label)
#         plt.scatter(data[indices, 0], data[indices, 1], c=[colors(label-1)], alpha=0.7, label=speakers[label-1])
    
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.title(title)
#     plt.grid(True)
#     plt.legend(loc='best', title='Speakers')
#     plt.tight_layout()
#     plt.savefig(output_file)
#     plt.show()
#     print(f"Scatter plot saved to: {output_file}")

# def process_and_plot(root_folder, output_plot_folder):
#     os.makedirs(output_plot_folder, exist_ok=True)
#     speakers = sorted(os.listdir(root_folder))
    
#     all_data = []
#     labels = []
    
#     for label, speaker in enumerate(speakers, start=1):
#         speaker_path = os.path.join(root_folder, speaker)
        
#         if os.path.isdir(speaker_path):
#             npy_files = list_all_npy_files(speaker_path)
            
#             for npy_file in npy_files:
#                 array = load_npy(npy_file)
#                 all_data.append(array)
#                 labels.append(label)  # Assign label to each speaker
    
#     if all_data:
#         all_data = np.vstack(all_data)
#         labels = np.array(labels)
#         print(f"Shape before PCA: {all_data.shape}")
        
#         reduced_data = apply_pca_and_get_2d_vectors(all_data)
#         print(f"Shape after PCA: {reduced_data.shape}")
        
#         output_file = os.path.join(output_plot_folder, "combined_scatter_plot.png")
#         plot_scatter(reduced_data, labels, output_file, "Combined Scatter Plot for Audio Files", speakers)
#     else:
#         print("No data to plot.")

# # Example usage:
# npy_folder_path = "/Users/sree/Desktop/ResultsFirstLayerCS(512X1 vectors)"
# output_plot_folder = "/Users/sree/Desktop/ScatterPlots"

# process_and_plot(npy_folder_path, output_plot_folder)