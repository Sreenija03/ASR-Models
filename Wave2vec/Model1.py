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

# def load_audio_files_from_folder(folder_path, sample_rate=16000):
#     audio_files = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.wav'):
#             filepath = os.path.join(folder_path, filename)
#             # Resampling to given sample sample rate
#             audio, sr = librosa.load(filepath, sr=sample_rate)
#             audio_files.append(audio)
#     return audio_files

# def process_batches(audio_files, processor, max_batch_size):
#     batch = audio_files[:max_batch_size]
#     # print(batch)
#     inputs = processor(batch, sampling_rate=16000, padding=True, return_tensors="pt")
#     return [inputs.input_values]

# def get_summed_outputs(model, batches):
#     all_outputs = []
#     for batch in batches:
#         outputs = model(batch)
#         all_outputs.append(outputs)
#     summed_outputs = torch.sum(torch.cat(all_outputs, dim=0), dim=2)
#     return summed_outputs.t()

# def calculate_cosine_similarity(outputs):
#     outputs_np = outputs.detach().numpy()  # Convert tensor to NumPy array
#     num_vectors = outputs_np.shape[1]
#     cosine_similarities_matrix = np.zeros((num_vectors, num_vectors))
#     for i in range(num_vectors):
#         for j in range(num_vectors):
#             cosine_similarities_matrix[i, j] = np.dot(outputs_np[:, i], outputs_np[:, j]) / (np.linalg.norm(outputs_np[:, i]) * np.linalg.norm(outputs_np[:, j]))
#     return cosine_similarities_matrix

# model = SimpleWav2Vec2CNN()

# folder_path = "/Users/sree/Desktop/Audio/Speaker2/Sent1"
# audio_files = load_audio_files_from_folder(folder_path)

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# max_batch_size = 10
# batches = process_batches(audio_files, processor, max_batch_size)
# print("Batches")
# print(len(batches))

# summed_outputs = get_summed_outputs(model, batches)

# cosine_similarities_matrix = calculate_cosine_similarity(summed_outputs)
# # print("Cosine Similarities Matrix:")
# # print(cosine_similarities_matrix)






import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

class SimpleWav2Vec2CNN(nn.Module):
    def __init__(self):
        super(SimpleWav2Vec2CNN, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def forward(self, input_values):
        outputs = self.wav2vec2.feature_extractor(input_values)
        return outputs

def load_audio_files_from_folder(folder_path, sample_rate=16000):
    audio_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            filepath = os.path.join(folder_path, filename)
            audio, sr = librosa.load(filepath, sr=sample_rate)
            audio_files.append(audio)
    return audio_files

def process_batches(audio_files, processor, max_batch_size):
    batches = []
    for i in range(0, len(audio_files), max_batch_size):
        batch = audio_files[i:i+max_batch_size]
        inputs = processor(batch, sampling_rate=16000, padding=True, return_tensors="pt")
        batches.append(inputs.input_values)
    return batches

def get_summed_outputs(model, batches):
    all_outputs = []
    for batch in batches:
        outputs = model(batch)
        all_outputs.append(outputs)
    summed_outputs = torch.sum(torch.cat(all_outputs, dim=0), dim=2)
    return summed_outputs.t()

def calculate_cosine_similarity(outputs):
    outputs_np = outputs.detach().numpy()  # Convert tensor to NumPy array
    num_vectors = outputs_np.shape[1]
    cosine_similarities_matrix = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            cosine_similarities_matrix[i, j] = np.dot(outputs_np[:, i], outputs_np[:, j]) / (np.linalg.norm(outputs_np[:, i]) * np.linalg.norm(outputs_np[:, j]))
    return cosine_similarities_matrix

model = SimpleWav2Vec2CNN()

root_folder = "/Users/sree/Desktop/Audio"
output_folder = "/Users/sree/Desktop/CosineMatrices"  # Folder to save output files
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
max_batch_size = 10

for speaker_folder in sorted(os.listdir(root_folder)):
    speaker_path = os.path.join(root_folder, speaker_folder)
    if os.path.isdir(speaker_path):
        print(f"Processing speaker: {speaker_folder}")
        speaker_output_folder = os.path.join(output_folder, speaker_folder)
        os.makedirs(speaker_output_folder, exist_ok=True)
        for subfolder in ['Sent1', 'Different']:
            subfolder_path = os.path.join(speaker_path, subfolder)
            if os.path.isdir(subfolder_path):
                print(f"  Processing {subfolder} folder")
                audio_files = load_audio_files_from_folder(subfolder_path)
                batches = process_batches(audio_files, processor, max_batch_size)
                summed_outputs = get_summed_outputs(model, batches)
                cosine_similarities_matrix = calculate_cosine_similarity(summed_outputs)
                output_file_path = os.path.join(speaker_output_folder, f"{subfolder}_cosine_similarity_matrix.txt")
                np.savetxt(output_file_path, cosine_similarities_matrix, fmt='%.6f')
                print(f"    Cosine Similarities Matrix for {subfolder} saved to: {output_file_path}")
