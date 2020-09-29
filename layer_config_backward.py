import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron_backward(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MultiLayerPerceptron_backward, self).__init__()
        #################################################################################
        # Initialize the modules required to implement the mlp with given layer   #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        #################################################################################
        layers = []
        layers.append(nn.Linear((input_size), (hidden_layers[0])))
        layers.append(nn.Linear((hidden_layers[0]), (hidden_layers[1])))
        layers.append(nn.Linear((hidden_layers[1]), (hidden_layers[2])))
        layers.append(nn.Linear((hidden_layers[2]), (hidden_layers[3])))
        layers.append(nn.Linear((hidden_layers[3]), (num_classes)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #################################################################################
        # Forward pass computations                                 #
        #################################################################################
        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        x = F.relu(self.layers[2](x))
        x = F.relu(self.layers[3](x))
        x = F.relu(self.layers[4](x))
        out = x
        return out