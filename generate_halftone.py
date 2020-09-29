import scipy.io
import numpy as np
import torch
from layer_config_backward import MultiLayerPerceptron_backward
from layer_config_forward import MultiLayerPerceptron_forward
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# Load the forward model
Forward_model = MultiLayerPerceptron_forward(8, [150, 150, 150], 31)
Forward_model.load_state_dict(torch.load('Forward_model.ckpt'))
Forward_model.to(device)
for param in Forward_model.parameters():
    param.requires_grad = False

# Load adaptive backward model
model_backward = MultiLayerPerceptron_backward(31, [300, 300, 200, 100, 200], 8)
model_backward.load_state_dict(torch.load('model_backward_flower.ckpt'))
model_backward.to(device)
for param in model_backward.parameters():
    param.requires_grad = False

# Load the spectral image to generate the halftone pattern
mat = scipy.io.loadmat('flower_painting_approx_spec.mat')
test_all=mat['flower_painting_approx_spec']

def RMSELoss(yhat,y):
    return torch.mean((torch.sqrt(torch.sum((yhat - y)**2,1))))/5.57*100 # RMSE loss

Reproduction_error = 0
for i in range(0, math.floor(test_all.shape[0]/500000)+1):
    test_spec=test_all[500000*i:500000*(i+1),:]
    test_spec_tensor = torch.from_numpy(test_spec)
    test_spec_tensor_cuda = test_spec_tensor.to(torch.device("cuda"))
    test_spec_tensor_cuda = test_spec_tensor_cuda.type(torch.cuda.FloatTensor)

    approx_halftone = model_backward(test_spec_tensor_cuda)
    Reproduction_error = RMSELoss(Forward_model(approx_halftone), test_spec_tensor_cuda) + Reproduction_error

    approx_halftone_cpu = approx_halftone.to(torch.device("cpu"))
    approx_halftone_cpu = approx_halftone_cpu.detach().numpy()

    path="./painting_halftone/painting%d" %(i)
    np.save(path,approx_halftone_cpu)

print('Painting average reproduction RMSE% = {} %'.format(Reproduction_error/(math.floor(test_all.shape[0]/500000)+1)))
