# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import librosa
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# import numpy as np


# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         outputs = self.wav2vec2.feature_extractor(input_values)
#         return outputs
 
# def Model_out(path_single_audio,sample_rate):
#     vector_audio, sr = librosa.load(path_single_audio, sr=sample_rate)
#     vector_audio=np.expand_dims(vector_audio,axis=0)
#     vector_tensor=torch.tensor(vector_audio,dtype=torch.float32)
#     # print(vector_tensor.shape)
#     outputs_tensor=model(vector_tensor)
#     # print(outputs_tensor.shape)
#     summed_outputs = torch.sum(outputs_tensor, dim=2)
#     summed_outputs=summed_outputs.detach().numpy()
#     return summed_outputs

# Main Code 
# import os
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# import matplotlib.pyplot as plt

# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         outputs = self.wav2vec2.feature_extractor(input_values)
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



# def compute_histograms_from_cosine_matrix(cosine_similarities_matrix, output_folder, category):
#     os.makedirs(output_folder, exist_ok=True)
#     n, m = cosine_similarities_matrix.shape
#     upper_indices = np.triu_indices(n=n, k=1, m=m)

#     matrix_values = cosine_similarities_matrix[upper_indices]
#     bins = np.linspace(0, 1, num=1000)

#     # Compute the histogram
#     histogram, bins = np.histogram(matrix_values, bins='auto')
#     histogram = histogram / matrix_values.size  # Normalize histogram

#     # Compute the cumulative frequency
#     cumulative_frequency = np.cumsum(histogram)
#     cumulative_frequency = cumulative_frequency / cumulative_frequency[-1]  # Normalize to 1

#     # Plot and save the histogram and cumulative frequency line
#     plt.figure()
#     plt.hist(bins[:-1], bins, weights=histogram, alpha=0.6, label='Histogram')
#     plt.plot(bins[:-1] + (bins[1] - bins[0]) / 2, cumulative_frequency, color='r', label='Cumulative Frequency')
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     plt.xlabel('Cosine Similarity')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram and Cumulative Frequency for {category}')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     output_file_path = os.path.join(output_folder, f"{category}_histogram_cumulative.jpg")
#     plt.savefig(output_file_path)
#     plt.close()
#     print(f"Histogram and Cumulative Frequency for {category} saved to: {output_file_path}")


# def compute_cosine_similarity_matrix(root_folder, output_folder):
#     for speaker_folder in sorted(os.listdir(root_folder)):
#         speaker_path = os.path.join(root_folder, speaker_folder)
#         if os.path.isdir(speaker_path):
#             print(f"Processing speaker: {speaker_folder}")
#             speaker_output_folder = os.path.join(output_folder, speaker_folder)
#             os.makedirs(speaker_output_folder, exist_ok=True)
#             for subfolder in ['Sent1', 'Different']:
#                 subfolder_path = os.path.join(speaker_path, subfolder)
#                 if os.path.isdir(subfolder_path):
#                     summed_outputs = load_summed_outputs(subfolder_path)
#                     cosine_similarities_matrix = cosine_similarity_matrix(summed_outputs, summed_outputs)
#                     output_file_path = os.path.join(speaker_output_folder, f"{subfolder}_cosine_similarity_matrix.txt")
#                     np.savetxt(output_file_path, cosine_similarities_matrix, fmt='%.6f')
#                     print(f"Cosine Similarities Matrix for {subfolder} saved to: {output_file_path}")

#                     # Compute and save histograms
#                     histogram_folder = os.path.join(speaker_output_folder, f"{subfolder}_histograms")
#                     compute_histograms_from_cosine_matrix(cosine_similarities_matrix, histogram_folder, subfolder)


# model = SimpleWav2Vec2CNN()
# folder_path = "/Users/sree/Desktop/Audio"
# output_folder = "/Users/sree/Desktop/Results2"
# out_folder="/Users/sree/Desktop/Cosine1"
# wav_files = list_all_wav_files(folder_path)
# # save_summed_outputs_as_npy(wav_files, 16000,output_folder)
# compute_cosine_similarity_matrix(output_folder, out_folder)








#Testing
# import os
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# import matplotlib.pyplot as plt

# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         outputs = self.wav2vec2.feature_extractor(input_values)
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

# def compute_histograms_from_cosine_matrix(cosine_similarities_matrix, output_folder, category):
#     os.makedirs(output_folder, exist_ok=True)
#     n, m = cosine_similarities_matrix.shape
#     upper_indices = np.triu_indices(n=n, k=1, m=m)

#     matrix_values = cosine_similarities_matrix[upper_indices]
#     bins = np.linspace(0, 1, num=1000)

#     # Compute the histogram
#     histogram, bins = np.histogram(matrix_values, bins='auto')
#     histogram = histogram / matrix_values.size

#     # Compute the cumulative frequency
#     cumulative_frequency = np.cumsum(histogram)
#     cumulative_frequency = cumulative_frequency / cumulative_frequency[-1]  # Normalize to 1

#     # Plot and save the histogram and cumulative frequency line
#     plt.figure()
#     plt.hist(bins[:-1], bins, weights=histogram, alpha=0.6, label='Histogram')
#     plt.plot(bins[:-1] + (bins[1] - bins[0]) / 2, cumulative_frequency, color='r', label='Cumulative Frequency')
#     plt.xlabel('Cosine Similarity')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram and Cumulative Frequency for {category}')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     output_file_path = os.path.join(output_folder, f"{category}_histogram_cumulative.jpg")
#     plt.savefig(output_file_path)
#     plt.close()
#     print(f"Histogram and Cumulative Frequency for {category} saved to: {output_file_path}")

# def compute_cosine_similarity_matrix_all_speakers(root_folder, output_folder, batch_size=10, max_files=100):
#   """
#   This function computes the cosine similarity matrix between all speakers in batches.

#   Args:
#       root_folder: Path to the folder containing speaker subfolders.
#       output_folder: Path to the folder where the final similarity matrix will be saved.
#       batch_size: Number of speaker outputs to process in each batch (default: 10).
#       max_files: Maximum number of files to process (default: 100).
#   """

#   all_summed_outputs = []
#   num_files_processed = 0
#   for speaker_folder in sorted(os.listdir(root_folder)):
#     speaker_path = os.path.join(root_folder, speaker_folder)
#     if os.path.isdir(speaker_path):
#       print(f"Processing speaker: {speaker_folder}")
#       for subfolder in ['Different']:
#         subfolder_path = os.path.join(speaker_path, subfolder)
#         if os.path.isdir(subfolder_path):
#           summed_outputs = load_summed_outputs(subfolder_path)
#           all_summed_outputs.extend(summed_outputs)
#           num_files_processed += len(summed_outputs)
#           if num_files_processed >= max_files:
#             break
#       if num_files_processed >= max_files:
#         break

#   all_summed_outputs = np.array(all_summed_outputs)[:max_files]
#   num_batches = len(all_summed_outputs) // batch_size + int(len(all_summed_outputs) % batch_size != 0)
#   final_cosine_similarities_matrix = np.zeros((max_files, max_files))

#   for batch_index in range(num_batches):
#     start_index = batch_index * batch_size
#     end_index = min((batch_index + 1) * batch_size, len(all_summed_outputs))
#     batch_outputs = all_summed_outputs[start_index:end_index]

#     batch_cosine_similarities = cosine_similarity_matrix(batch_outputs, all_summed_outputs)

#     # Update the final matrix section corresponding to the current batch
#     final_cosine_similarities_matrix[start_index:end_index, :] = batch_cosine_similarities[:, :max_files]  # Assuming max_files represents total speakers


#   # Save the final cosine similarity matrix
#   final_output_file_path = os.path.join(output_folder, "all_speakers_cosine_similarity_matrix.txt")
#   np.savetxt(final_output_file_path, final_cosine_similarities_matrix, fmt='%.6f')
#   print(f"Final Cosine Similarities Matrix for all speakers saved to: {final_output_file_path}")


# # Modify the paths accordingly
# folder_path = "/Users/sree/Desktop/Audio"
# output_folder = "/Users/sree/Desktop/Results"
# out_folder = "/Users/sree/Desktop/Cosine2"

# # Compute cosine similarity matrix for all speakers
# compute_cosine_similarity_matrix_all_speakers(output_folder, out_folder, batch_size=10)



#Testing2(ACLR Loss)Adaptive Contrastive Loss with Content-preserving Regularization
# import os
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from transformers import Wav2Vec2Model
# import matplotlib.pyplot as plt


# class Wav2Vec2SpeakerDisentanglement(nn.Module):
#     def __init__(self, wav2vec2_model, projection_dim=256):
#         super(Wav2Vec2SpeakerDisentanglement, self).__init__()
#         self.wav2vec2 = wav2vec2_model
#         self.projection_head = nn.Linear(self.wav2vec2.config.hidden_size, projection_dim)

#     def forward(self, input_values):
#         outputs = self.wav2vec2(input_values)
#         encoded_features = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

#         # Calculate contrastive loss using cosine similarity
#         contrastive_loss = self.calculate_contrastive_loss(encoded_features)

#         # Optionally, add content-preserving regularization term (choose one)
#         content_loss = self.calculate_content_loss(encoded_feZzatures)

#         # Combine losses with weighting factor (lambda)
#         lambda_contrastive = 1.0
#         lambda_content = 0.1
#         total_loss = lambda_contrastive * contrastive_loss + lambda_content * content_loss

#         # Return speaker-normalized features and total loss
#         speaker_normalized_features = self.projection_head(encoded_features)
#         return speaker_normalized_features, total_loss

#     def calculate_contrastive_loss(self, encoded_features):
#         # Calculate cosine similarity between all pairs of encoded features
#         similarity_matrix = torch.cosine_similarity(encoded_features, encoded_features)
#         # Calculate contrastive loss using the similarity matrix
#         contrastive_loss = torch.mean(similarity_matrix * (1 - torch.eye(similarity_matrix.shape[0])))
#         return contrastive_loss

#     def calculate_content_loss(self, encoded_features):
#         # Calculate content loss using a simple L2 regularization term
#         content_loss = torch.mean(torch.norm(encoded_features, dim=-1))
#         return content_loss

# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         outputs = self.wav2vec2(input_values)
#         return outputs.last_hidden_state

# def model_out(path_single_audio, sample_rate, model):
#     vector_audio, sr = librosa.load(path_single_audio, sr=sample_rate)
#     vector_audio = np.expand_dims(vector_audio, axis=0)
#     vector_tensor = torch.tensor(vector_audio, dtype=torch.float32)
#     with torch.no_grad():
#         outputs_tensor = model(vector_tensor)
#     summed_outputs = torch.sum(outputs_tensor, dim=1)  # Summing across the sequence dimension
#     summed_outputs = summed_outputs.detach().numpy()
#     return summed_outputs

# def list_all_wav_files(root_folder):
#     wav_files = []
#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             if filename.endswith('.wav'):
#                 wav_files.append(os.path.join(dirpath, filename))
#     return wav_files

# def save_summed_outputs_as_npy(files, sample_rate, output_root_folder, model):
#     for file in files:
#         summed_outputs = model_out(file, sample_rate, model)
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

# def save_matrix(matrix, output_folder, label):
#     output_path = os.path.join(output_folder, f'cosine_similarity_matrix_{label}.txt')
#     if not os.path.exists(output_path):
#         np.savetxt(output_path, matrix, fmt='%.6f')

# def compute_histograms_from_cosine_matrix(cosine_similarities_matrix, output_folder, category):
#     os.makedirs(output_folder, exist_ok=True)
#     n, m = cosine_similarities_matrix.shape
#     upper_indices = np.triu_indices(n=n, k=1, m=m)

#     matrix_values = cosine_similarities_matrix[upper_indices]
#     bins = np.linspace(0, 1, num=1000)

#     # Compute the histogram
#     histogram, bins = np.histogram(matrix_values, bins='auto')
#     histogram = histogram / matrix_values.size  # Normalize histogram

#     # Compute the cumulative frequency
#     cumulative_frequency = np.cumsum(histogram)
#     cumulative_frequency = cumulative_frequency / cumulative_frequency[-1]  # Normalize to 1

#     # Plot and save the histogram and cumulative frequency line
#     plt.figure()
#     plt.hist(bins[:-1], bins, weights=histogram, alpha=0.6, label='Histogram')
#     plt.plot(bins[:-1] + (bins[1] - bins[0]) / 2, cumulative_frequency, color='r', label='Cumulative Frequency')
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     plt.xlabel('Cosine Similarity')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram and Cumulative Frequency for {category}')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     output_file_path = os.path.join(output_folder, f"{category}_histogram_cumulative.jpg")
#     plt.savefig(output_file_path)
#     plt.close()
#     print(f"Histogram and Cumulative Frequency for {category} saved to: {output_file_path}")

# def compute_cosine_similarity_matrix_combined(root_folder, output_folder):
#     combined_summed_outputs_sent1 = []
#     combined_summed_outputs_different = []

#     for speaker_folder in sorted(os.listdir(root_folder)):
#         speaker_path = os.path.join(root_folder, speaker_folder)
#         if os.path.isdir(speaker_path):
#             print(f"Processing speaker: {speaker_folder}")
#             for subfolder in ['Sent1', 'Different']:
#                 subfolder_path = os.path.join(speaker_path, subfolder)
#                 if os.path.isdir(subfolder_path):
#                     summed_outputs = load_summed_outputs(subfolder_path)
#                     if subfolder == 'Sent1':
#                         combined_summed_outputs_sent1.append(summed_outputs)
#                     else:
#                         combined_summed_outputs_different.append(summed_outputs)
    
#     # Combine all summed outputs for each category
#     if combined_summed_outputs_sent1:
#         combined_summed_outputs_sent1 = np.concatenate(combined_summed_outputs_sent1, axis=0)
#         cosine_similarities_matrix_sent1 = cosine_similarity_matrix(combined_summed_outputs_sent1, combined_summed_outputs_sent1)
#         save_matrix(cosine_similarities_matrix_sent1, output_folder, 'Sent1')
#         compute_histograms_from_cosine_matrix(cosine_similarities_matrix_sent1, output_folder, 'Sent1')
    
#     if combined_summed_outputs_different:
#         combined_summed_outputs_different = np.concatenate(combined_summed_outputs_different, axis=0)
#         cosine_similarities_matrix_different = cosine_similarity_matrix(combined_summed_outputs_different, combined_summed_outputs_different)
#         save_matrix(cosine_similarities_matrix_different, output_folder, 'Different')
#         compute_histograms_from_cosine_matrix(cosine_similarities_matrix_different, output_folder, 'Different')

# # Initialize models and paths
# wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
# model = SimpleWav2Vec2CNN()
# folder_path = "/Users/sree/Desktop/Audio"
# output_folder = "/Users/sree/Desktop/Results2"
# out_folder = "/Users/sree/Desktop/Cosine1"

# # List all wav files in the specified folder
# wav_files = list_all_wav_files(folder_path)
# # save_summed_outputs_as_npy(wav_files, 16000, output_folder,model)

# # Compute cosine similarity matrix and save results
# compute_cosine_similarity_matrix_combined(output_folder, out_folder)









# Spectrogram generation for one audio file
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np

# def generate_spectrogram(audio_path, output_path):
#     # Load the audio file
#     y, sr = librosa.load(audio_path, sr=None)
    
#     # Compute the spectrogram
#     D = librosa.stft(y)
#     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
#     # Plot the spectrogram
#     plt.figure(figsize=(10, 6))
#     librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Spectrogram')
    
#     # Save the plot to a file
#     plt.savefig(output_path)
#     plt.close()

# # Example usage
# audio_path = '/Users/sree/Desktop/Audio/Speaker9/Sent1/Audio1.wav'
# output_path = '/Users/sree/Desktop/Audio1_spectrogram.png'
# generate_spectrogram(audio_path, output_path)





# # Least Cosine similarity values in the matrix along with their indices

# import numpy as np

# def read_cosine_matrix(file_path):
#     matrix = np.loadtxt(file_path)
#     return matrix

# def find_least_values_with_indices(matrix, num_values=10):
#     flat_matrix = matrix.flatten()

#     mask = np.eye(matrix.shape[0], dtype=bool).flatten()
#     flat_matrix = np.where(mask, np.inf, flat_matrix)
#     least_indices_flat = np.argsort(flat_matrix)[:num_values]
    
#     least_values = flat_matrix[least_indices_flat]

#     least_indices_2d = [np.unravel_index(idx, matrix.shape) for idx in least_indices_flat]
    
#     return least_values, least_indices_2d

# file_path = '/Users/sree/Desktop/Cosine3/cosine_similarity_matrix_Sent1.txt'
# matrix = read_cosine_matrix(file_path)
# least_values, least_indices = find_least_values_with_indices(matrix)

# print("Least 10 values in the cosine similarity matrix and their indices:")
# for value, index in zip(least_values, least_indices):
#     print(f"Value: {value}, Index: {index}")









# import os
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt

# def read_cosine_matrix(file_path):
#     # Read the matrix from the text file
#     matrix = np.loadtxt(file_path)
#     return matrix

# def find_least_values_with_indices(matrix, num_values=10):
#     # Flatten the matrix
#     flat_matrix = matrix.flatten()
    
#     # Exclude the diagonal values (self-similarity, which are 1.0 or close to it)
#     mask = np.eye(matrix.shape[0], dtype=bool).flatten()
#     flat_matrix = np.where(mask, np.inf, flat_matrix)
    
#     # Get the indices of the least values
#     least_indices_flat = np.argsort(flat_matrix)[:num_values]
    
#     # Get the actual values
#     least_values = flat_matrix[least_indices_flat]
    
#     # Convert flat indices back to 2D indices
#     least_indices_2d = [np.unravel_index(idx, matrix.shape) for idx in least_indices_flat]
    
#     return least_values, least_indices_2d

# def generate_and_save_spectrogram(audio_path, output_folder, index):
#     y, sr = librosa.load(audio_path)
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     S_dB = librosa.power_to_db(S, ref=np.max)

#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Mel-frequency spectrogram')
#     plt.tight_layout()

#     # Save the spectrogram
#     spectrogram_path = os.path.join(output_folder, f'spectrogram_{index}.png')
#     plt.savefig(spectrogram_path)
#     plt.close()

#     print(f'Saved spectrogram for index {index} at {spectrogram_path}')

# def create_spectrograms_for_indices(indices, audio_root_folder, output_root_folder):
#     os.makedirs(output_root_folder, exist_ok=True)
    
#     for idx, index in enumerate(indices):
#         # Parse the speaker number and audio file number from the index
#         speaker_num = str(index[0] // 10).zfill(2)  # Assuming speaker numbers are zero-padded to two digits
#         audio_file_num = str(index[0] % 10)  # Assuming the audio file number is the last digit

#         # Construct the path to the audio file with "Sent1" included
#         audio_path = os.path.join(audio_root_folder, f'speaker{speaker_num}', 'Sent1', f'audio{audio_file_num}.wav')
        
#         if os.path.exists(audio_path):
#             # Create a new folder for each index
#             output_folder = os.path.join(output_root_folder, f'speaker{speaker_num}_audio{audio_file_num}')
#             os.makedirs(output_folder, exist_ok=True)
            
#             generate_and_save_spectrogram(audio_path, output_folder, f'{speaker_num}_{audio_file_num}')
#         else:
#             print(f'Audio file not found: {audio_path}')

# # Example usage
# file_path = '/Users/sree/Desktop/Model2/cosine_similarity_matrix_Sent1.txt'
# audio_root_folder = '/Users/sree/Desktop/Audio'
# output_root_folder = '/Users/sree/Desktop/Spectrograms2'
# num_values = 120

# matrix = read_cosine_matrix(file_path)
# least_values, least_indices = find_least_values_with_indices(matrix, num_values)

# print("Least 10 values in the cosine similarity matrix and their indices:")
# for value, index in zip(least_values, least_indices):
#     print(f"Value: {value}, Index: {index}")

# create_spectrograms_for_indices(least_indices, audio_root_folder, output_root_folder)



#Block Diagnols
# import os
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from transformers import Wav2Vec2Model
# import matplotlib.pyplot as plt

# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         outputs = self.wav2vec2.feature_extractor(input_values)
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

# def save_matrix(matrix, output_folder, label):
#     os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
#     file_path = os.path.join(output_folder, f'cosine_similarity_matrix_{label}.txt')
#     print(f"Saving matrix to: {file_path}")
#     try:
#         np.savetxt(file_path, matrix, fmt='%.6f')
#         print(f"Matrix successfully saved to: {file_path}")
#     except Exception as e:
#         print(f"Error saving matrix to file: {e}")

# def remove_block_diagonals(matrix, block_size):
#     n = matrix.shape[0]
#     for i in range(0, n, block_size):
#         for j in range(block_size):
#             if i + j < n:
#                 matrix[i+j, i:i+block_size] = 0
#                 matrix[i:i+block_size, i+j] = 0
#     return matrix


# def get_upper_triangular_without_blocks(matrix, block_size):
#     n = matrix.shape[0]
#     upper_triangular = np.triu(matrix)
#     for i in range(0, n, block_size):
#         for j in range(block_size):
#             if i + j < n:
#                 upper_triangular[i+j, i:i+block_size] = 0
#                 upper_triangular[i:i+block_size, i+j] = 0
#     return upper_triangular


# def save_upper_part_without_blocks(matrix, output_folder, label):
#     os.makedirs(output_folder, exist_ok=True) 
#     upper_part_matrix = get_upper_triangular_without_blocks(matrix, block_size=10)
#     file_path = os.path.join(output_folder, f'cosine_similarity_upper_part_{label}.txt')
#     print(f"Saving upper part without blocks matrix to: {file_path}")
#     try:
#         np.savetxt(file_path, upper_part_matrix, fmt='%.6f')
#         print(f"Upper part without blocks matrix successfully saved to: {file_path}")
#     except Exception as e:
#         print(f"Error saving upper part without blocks matrix to file: {e}")

# def compute_histograms_from_cosine_matrix(cosine_similarities_matrix, output_folder, category):
#     os.makedirs(output_folder, exist_ok=True)
    
#     block_size = 10  
#     cosine_similarities_matrix = remove_block_diagonals(cosine_similarities_matrix, block_size)
#     upper_triangular_matrix = get_upper_triangular_without_blocks(cosine_similarities_matrix, block_size)

#     matrix_values =upper_triangular_matrix[upper_triangular_matrix != 0]

#     bins = np.linspace(0, 1, num=1000)
#     histogram, bins = np.histogram(matrix_values, bins='auto')
#     histogram = histogram / cosine_similarities_matrix.size  # Normalize histogram
#     cumulative_frequency = np.cumsum(histogram)
#     cumulative_frequency = cumulative_frequency / cumulative_frequency[-1]  # Normalize to 1


#     plt.figure()
#     plt.hist(bins[:-1], bins, weights=histogram, label='Histogram')
#     plt.plot(bins[:-1] + (bins[1] - bins[0]) / 2, cumulative_frequency, color='r', label='Cumulative Frequency')
#     plt.xlim((0.5, 1))
#     plt.ylim((0, 1))
#     plt.xlabel('Cosine Similarity')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram and Cumulative Frequency for {category}')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     output_file_path = os.path.join(output_folder, f"{category}_histogram_cumulative.jpg")
#     plt.savefig(output_file_path)
#     plt.close()
#     print(f"Histogram and Cumulative Frequency for {category} saved to: {output_file_path}")

# def compute_cosine_similarity_matrix_combined(root_folder, output_folder):
#     combined_summed_outputs_sent1 = []
#     combined_summed_outputs_different = []

#     for speaker_folder in sorted(os.listdir(root_folder)):
#         speaker_path = os.path.join(root_folder, speaker_folder)
#         if os.path.isdir(speaker_path):
#             print(f"Processing speaker: {speaker_folder}")
#             for subfolder in ['Sent1', 'Different']:
#                 subfolder_path = os.path.join(speaker_path, subfolder)
#                 if os.path.isdir(subfolder_path):
#                     summed_outputs = load_summed_outputs(subfolder_path)
#                     if subfolder == 'Sent1':
#                         combined_summed_outputs_sent1.append(summed_outputs)
#                     else:
#                         combined_summed_outputs_different.append(summed_outputs)
    
#     if combined_summed_outputs_sent1:
#         combined_summed_outputs_sent1 = np.concatenate(combined_summed_outputs_sent1, axis=0)
#         cosine_similarities_matrix_sent1 = cosine_similarity_matrix(combined_summed_outputs_sent1, combined_summed_outputs_sent1)
#         save_matrix(cosine_similarities_matrix_sent1, output_folder, 'Sent1')
#         save_upper_part_without_blocks(cosine_similarities_matrix_sent1, output_folder, 'Sent1')
#         compute_histograms_from_cosine_matrix(cosine_similarities_matrix_sent1, output_folder, 'Sent1')
    
#     if combined_summed_outputs_different:
#         combined_summed_outputs_different = np.concatenate(combined_summed_outputs_different, axis=0)
#         cosine_similarities_matrix_different = cosine_similarity_matrix(combined_summed_outputs_different, combined_summed_outputs_different)
#         save_matrix(cosine_similarities_matrix_different, output_folder, 'Different')
#         save_upper_part_without_blocks(cosine_similarities_matrix_different, output_folder, 'Different')
#         compute_histograms_from_cosine_matrix(cosine_similarities_matrix_different, output_folder, 'Different')

# # Ensure the model is correctly initialized
# model = SimpleWav2Vec2CNN()
# folder_path = "/Users/sree/Desktop/Audio"
# output_folder = "/Users/sree/Desktop/Results"
# out_folder = "/Users/sree/Desktop/Cosine5"
# wav_files = list_all_wav_files(folder_path)

# # Uncomment this line if you need to process and save .npy files first
# # save_summed_outputs_as_npy(wav_files, 16000, output_folder)

# compute_cosine_similarity_matrix_combined(output_folder, out_folder)




# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import librosa
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# import numpy as np


# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         outputs = self.wav2vec2.feature_extractor(input_values)
#         return outputs
 
# def Model_out(path_single_audio,sample_rate):
#     vector_audio, sr = librosa.load(path_single_audio, sr=sample_rate)
#     vector_audio=np.expand_dims(vector_audio,axis=0)
#     vector_tensor=torch.tensor(vector_audio,dtype=torch.float32)
#     # print(vector_tensor.shape)
#     outputs_tensor=model(vector_tensor)
#     # print(outputs_tensor.shape)
#     summed_outputs = torch.sum(outputs_tensor, dim=2)
#     summed_outputs=summed_outputs.detach().numpy()
#     return summed_outputs





# import os
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from transformers import Wav2Vec2Processor, Wav2Vec2Model
# import matplotlib.pyplot as plt

# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         outputs = self.wav2vec2.feature_extractor(input_values)
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

#     # Set diagonal and lower triangular elements to 0
#     np.fill_diagonal(out, 0)
#     out = np.triu(out, k=1)  # Set lower triangular part (excluding diagonal) to 0

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



# def compute_histograms_from_cosine_matrix(cosine_similarities_matrix, output_folder, category):
#     os.makedirs(output_folder, exist_ok=True)
#     n, m = cosine_similarities_matrix.shape
#     upper_indices = np.triu_indices(n=n, k=1, m=m)

#     matrix_values = cosine_similarities_matrix[upper_indices]
#     bins = np.linspace(0, 1, num=1000)

#     # Compute the histogram
#     histogram, bins = np.histogram(matrix_values, bins='auto')
#     histogram = histogram / matrix_values.size  # Normalize histogram

#     # Compute the cumulative frequency
#     cumulative_frequency = np.cumsum(histogram)
#     cumulative_frequency = cumulative_frequency / cumulative_frequency[-1]  # Normalize to 1

#     # Plot and save the histogram and cumulative frequency line
#     plt.figure()
#     plt.hist(bins[:-1], bins, weights=histogram, alpha=0.6, label='Histogram')
#     plt.plot(bins[:-1] + (bins[1] - bins[0]) / 2, cumulative_frequency, color='r', label='Cumulative Frequency')
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     plt.xlabel('Cosine Similarity')
#     plt.ylabel('Frequency')
#     plt.title(f'Histogram and Cumulative Frequency for {category}')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     output_file_path = os.path.join(output_folder, f"{category}_histogram_cumulative.jpg")
#     plt.savefig(output_file_path)
#     plt.close()
#     print(f"Histogram and Cumulative Frequency for {category} saved to: {output_file_path}")


# def compute_cosine_similarity_matrix(root_folder, output_folder):
#     for speaker_folder in sorted(os.listdir(root_folder)):
#         speaker_path = os.path.join(root_folder, speaker_folder)
#         if os.path.isdir(speaker_path):
#             print(f"Processing speaker: {speaker_folder}")
#             speaker_output_folder = os.path.join(output_folder, speaker_folder)
#             os.makedirs(speaker_output_folder, exist_ok=True)
#             for subfolder in ['Sent1', 'Different']:
#                 subfolder_path = os.path.join(speaker_path, subfolder)
#                 if os.path.isdir(subfolder_path):
#                     summed_outputs = load_summed_outputs(subfolder_path)
#                     cosine_similarities_matrix = cosine_similarity_matrix(summed_outputs, summed_outputs)
#                     output_file_path = os.path.join(speaker_output_folder, f"{subfolder}_cosine_similarity_matrix.txt")
#                     np.savetxt(output_file_path, cosine_similarities_matrix, fmt='%.6f')
#                     print(f"Cosine Similarities Matrix for {subfolder} saved to: {output_file_path}")

#                     # Compute and save histograms
#                     histogram_folder = os.path.join(speaker_output_folder, f"{subfolder}_histograms")
#                     compute_histograms_from_cosine_matrix(cosine_similarities_matrix, histogram_folder, subfolder)


# model = SimpleWav2Vec2CNN()
# folder_path = "/Users/sree/Desktop/Audio"
# output_folder = "/Users/sree/Desktop/Results"
# out_folder="/Users/sree/Desktop/Cosine6"
# wav_files = list_all_wav_files(folder_path)
# # save_summed_outputs_as_npy(wav_files, 16000,output_folder)
# compute_cosine_similarity_matrix(output_folder, out_folder)









