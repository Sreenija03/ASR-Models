
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class SimpleWav2Vec2CNN(nn.Module):
    def __init__(self):
        super(SimpleWav2Vec2CNN, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def forward(self, input_values):
        outputs = self.wav2vec2.feature_extractor(input_values)
        return outputs

def load_audio_files_from_folder(folder_path, sample_rate=16000):
    audio_files = []
    max_length = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            filepath = os.path.join(folder_path, filename)
            audio, sr = librosa.load(filepath, sr=sample_rate)
            if len(audio) > max_length:
                max_length = len(audio)
            audio_files.append(audio)
    return audio_files, max_length

def process_batches(audio_files, processor, batch_size):
    batch = audio_files[:batch_size]
    inputs = processor(batch, sampling_rate=16000, padding=True, return_tensors="pt")
    return inputs.input_values

def get_summed_outputs(model, batch_tensor):
    with torch.no_grad():
        outputs = model(batch_tensor)
    outputs_numpy = outputs.cpu().numpy()
    summed_outputs = np.sum(outputs_numpy, axis=1)
    return summed_outputs

def calculate_cosine_similarity(outputs):
    num_vectors = outputs.shape[0]
    cosine_similarities_matrix = np.zeros((num_vectors, num_vectors))
    for i in range(num_vectors):
        for j in range(num_vectors):
            cosine_similarities_matrix[i, j] = np.dot(outputs[i], outputs[j]) / (np.linalg.norm(outputs[i]) * np.linalg.norm(outputs[j]))
    return cosine_similarities_matrix

def main():
    model = SimpleWav2Vec2CNN()
    folder_path = "/Users/sree/Downloads/archive (1)/fold1"
    audio_files, max_length = load_audio_files_from_folder(folder_path)


    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    max_batch_size = 10

    first_batch = process_batches(audio_files, processor, max_batch_size)
    # print(first_batch.size())
    print(max_length)
    summed_outputs = get_summed_outputs(model, first_batch)
    # print(summed_outputs.size())

    cosine_similarities_matrix = calculate_cosine_similarity(summed_outputs)

    print("Cosine Similarities Matrix:")
    print(cosine_similarities_matrix)

if __name__ == "__main__":
    main()
