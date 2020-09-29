import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from layer_config_forward import MultiLayerPerceptron_forward
import scipy.io
import numpy as np
from layer_config_backward import MultiLayerPerceptron_backward

def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# --------------------------------
# Device configuration
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)
# --------------------------------
# Hyper-parameters
# --------------------------------
input_size = 31
hidden_size = [300, 300, 200, 100, 200]
num_classes = 8
num_epochs = 20
batch_size = 200
learning_rate = 5 * 1e-3
learning_rate_decay = 0.95
reg = 0.001
num_training = 49000
num_validation = 1000
train = True  # False



mat = scipy.io.loadmat('flower_painting_approx_spec.mat')
SpecData = mat['flower_painting_approx_spec']


SpecData_samples = SpecData[np.random.choice(4612992,33945),:]


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


x_train_tensor = torch.from_numpy(SpecData_samples).float()
dataset = TensorDataset(x_train_tensor)
lengths = [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]

train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=31)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=31)





# Load the forward model
Forward_model = MultiLayerPerceptron_forward(8, [150, 150, 150], 31)
Forward_model.load_state_dict(torch.load('Forward_model.ckpt'))
Forward_model.to(device)
for param in Forward_model.parameters():
    param.requires_grad = False

# Load general backward model
model_backward = MultiLayerPerceptron_backward(input_size,hidden_size, num_classes)
model_backward.load_state_dict(torch.load('model_backward.ckpt'))
model_backward.to(device)
for param in model_backward.parameters():
    param.requires_grad = True

# Loss and optimizer
def RMSELoss(yhat,y):
    return torch.mean((torch.sqrt(torch.sum((yhat - y)**2,1))))/5.57*100 # RMSE loss

criterion_RMSE = RMSELoss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_backward.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model_backward
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, spectra in enumerate(train_loader):
        # Move tensors to the configured device
        spectra = torch.FloatTensor(spectra[0])
        spectra = spectra.to(device)
        #################################################################################
        #  Training                                             #
        ################################################################################
        optimizer.zero_grad()
        spec = spectra
        halftone_out = (model_backward(spec))
        outputs = Forward_model(halftone_out)
        loss = criterion_RMSE(outputs, spectra) + 0.02 * torch.norm(model_backward(spectra), p=1)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    with torch.no_grad():
        correct = 0
        total = 0
        spec_all = torch.zeros(len(val_loader), 31).to(device)
        outputs_all = torch.zeros(len(val_loader), 31).to(device)
        ####################################################
        # Evaluation
        for spec in val_loader:
            spec = torch.FloatTensor(spec[0])
            spec = spec.to(device)
            outputs = Forward_model(model_backward(spec))
            spec_all = torch.cat((spec_all, spec), 0)
            outputs_all = torch.cat((outputs_all, outputs), 0)

        loss_val = RMSELoss(outputs_all, spec_all)
        print('spec-spec loss is : {} %'.format(loss_val))
        print('area coverage loss is : {} %'.format(torch.norm(model_backward(spec), p=1)))

##################################################################################
# Save the model_backward_xxx checkpoint
torch.save(model_backward.state_dict(), 'model_backward_flower.ckpt')
