import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        resnet = models.resnet50(pretrained=True)
        
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, embed_size)
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        
    def forward(self, features, captions):
        embedded = self.embed(captions)
        embedded = torch.cat((features.unsqueeze(dim = 1), embedded), dim = 1)
        hidden, state = self.lstm(embedded)
        
        output = self.fc(hidden)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predictions = []
        embeddings = inputs
        
        for l in range(max_len):
            hidden, state = self.lstm(inputs, states)
            output = self.fc(hidden.squeeze(1))
            _, prediction = torch.max(output, 1)
            predictions.append(prediction.item())
            inputs = self.embed(prediction).unsqueeze(1)
        
        
        return predictions
       
            
          
        
