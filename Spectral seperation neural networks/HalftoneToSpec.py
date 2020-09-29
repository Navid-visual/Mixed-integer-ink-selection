import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 8
hidden_size = [150,150,150]
num_classes = 31
num_epochs = 19
batch_size = 200
learning_rate = 1e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
train =True#True
import scipy.io


mat = scipy.io.loadmat('dataset_approx_spec.mat')
SpecData=mat['dataset_approx_spec']

mat = scipy.io.loadmat('dataset_halftone.mat')
HalftoneData=mat['dataset_halftone']

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)
    

x_train_tensor = torch.from_numpy(SpecData).float()        
y_train_tensor = torch.from_numpy(HalftoneData).float()

dataset  = TensorDataset(x_train_tensor, y_train_tensor)


lengths = [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)]

train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=31)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=31)


class MultiLayerPerceptron_forward(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(MultiLayerPerceptron_forward, self).__init__()
        #################################################################################
        # Initialize the modules required to implement the mlp with given layer   #
        # configuration. input_size --> hidden_layers[0] --> hidden_layers[1] .... -->  #
        # hidden_layers[-1] --> num_classes                                             #
        #################################################################################
        layers = []
        layers.append(nn.Linear((input_size), (hidden_layers[0])))
        layers.append(nn.Linear((hidden_layers[0]), (hidden_layers[1])))
        layers.append(nn.Linear((hidden_layers[1]), (hidden_layers[2])))
        layers.append(nn.Linear((hidden_layers[2]), (num_classes)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #################################################################################
        # Implement the forward pass computations                                 #
        #################################################################################

        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        x = F.relu(self.layers[2](x))
        x = F.relu(self.layers[3](x))
        
        out=x

        return out

model_HalftoneTospec = MultiLayerPerceptron_forward(input_size, hidden_size, num_classes).to(device)

model_HalftoneTospec.apply(weights_init)


# Loss and optimizer
def RMSELoss(yhat,y):
    return torch.mean((torch.sqrt(torch.sum((yhat - y)**2,1))))/5.57*100

criterion_RMSE = RMSELoss
optimizer = torch.optim.Adam(model_HalftoneTospec.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model_HalftoneTospec
lr = learning_rate
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (spec, halftone) in enumerate(train_loader):
        # Move tensors to the configured device
        halftone = halftone.to(device)
        spec = spec.to(device)
        #################################################################################
        # Implement the training code                                             #
        optimizer.zero_grad()
        im = halftone.view(31, input_size)
        outputs = model_HalftoneTospec(im)
        loss = criterion_RMSE(outputs, spec)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    with torch.no_grad():
        correct = 0
        total = 0
        spec_all = torch.zeros(len(val_loader),31).to(device)
        outputs_all = torch.zeros(len(val_loader),31).to(device)
        for spec, halftone in val_loader:
            spec = spec.to(device)
            spec_all = torch.cat((spec_all,spec),0)
            ####################################################
            #evaluation #
            halftone = halftone.to(device)
            outputs = model_HalftoneTospec(halftone.view(31, input_size))
            outputs_all = torch.cat((outputs_all,outputs),0)
        loss = criterion_RMSE(outputs_all, spec_all)
        print('Validataion RMSE% is: {} %'.format(loss))

# save the model
torch.save(model_HalftoneTospec.state_dict(), 'Forward_model.ckpt')




