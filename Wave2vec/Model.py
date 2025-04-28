###Different Models that we are using###

#Model for extracting the feature extractor layer

# import torch
# import torch.nn as nn
# from transformers import Wav2Vec2Model

# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
#     def forward(self, input_values):
#         outputs = self.wav2vec2.feature_extractor(input_values)
#         return outputs

# model = SimpleWav2Vec2CNN()
# input_values = torch.randn(1, 16000)
# outputs = model(input_values)
# print(outputs)
# print(outputs.shape)


#Model for extracting the 1st layer of feature extractor
# import torch
# import torch.nn as nn
# from transformers import Wav2Vec2Model

# class SimpleWav2Vec2CNN(nn.Module):
#     def __init__(self):
#         super(SimpleWav2Vec2CNN, self).__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#     def forward(self, input_values):
#         first_cnn_layer = self.wav2vec2.feature_extractor.conv_layers[0]
    
#         output = first_cnn_layer(input_values)
    
#         return output

# model = SimpleWav2Vec2CNN()
# input_values = torch.randn(1, 1, 16000)  
# outputs = model(input_values)
# print(outputs)
# print(outputs.shape)


#HUBert
# import torch
# import torch.nn as nn
# from transformers import HubertModel

# class SimpleHuBERTCNN(nn.Module):
#     def __init__(self):
#         super(SimpleHuBERTCNN, self).__init__()
#         self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

#     def forward(self, input_values):
#         # Ensure input_values has shape [num_channels, sequence_length]
#         input_values = input_values.squeeze(0)  # Remove the batch dimension if batch size is 1
#         outputs = self.hubert(input_values)
#         return outputs.last_hidden_state  # Assuming you want to use the last hidden state

# # Instantiate the model
# model = SimpleHuBERTCNN()

# # Example input values (single sample, single channel, 16000 samples)
# input_values = torch.randn(1, 1, 16000)

# # Get model outputs
# outputs = model(input_values)

# # Print outputs and shape
# print(outputs)
# print(outputs.shape)



import torch
import torch.nn as nn
from transformers import HubertModel

class HuBERTContentModel(nn.Module):
    def __init__(self):
        super(HuBERTContentModel, self).__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    def forward(self, input_values):
        # Forward pass through HuBERT
        input_values = input_values.squeeze(0)
        outputs = self.hubert(input_values)
        
        # Access last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Assuming quantization layer is applied after last hidden state
        # Example: Apply a simple linear layer for quantization
        quantized_features = nn.Linear(last_hidden_state.shape[-1], 512)(last_hidden_state)

        return quantized_features

# Instantiate the model
model = HuBERTContentModel()

# Example input values (single sample, single channel, 16000 samples)
input_values = torch.randn(1, 1, 16000)

# Get model outputs (quantized features)
outputs = model(input_values)

# Print outputs and shape
print(outputs)
print(outputs.shape)

