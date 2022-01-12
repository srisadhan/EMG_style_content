import torch
import torch.optim as optim

from utils import *
import yaml 
from pathlib import Path 
from StyleEncoder import StyleEncoderDataset, StyleEncoder

# Read the configurations from the config.yml file
config_path = Path(__file__).parents[1] / 'config.yml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader) 

def test_StyleEncoder(data):
    model = StyleEncoder()
    print(data['features'][0:6, :, :].shape)
    embeds = model(torch.tensor(data['features'][0:6, :, :], dtype=torch.float32))
    print(embeds.shape)

    loss, eer = model.total_loss(embeds.reshape(3, 2, 256), data['labels'][0:6])
    print(eer)

    loss.backward()
    
if __name__ == '__main__':
    train_datapath = Path(__file__).parents[1] / 'data/processed/Train.h5'
    test_datapath  = Path(__file__).parents[1] / 'data/processed/Test.h5'
    valid_datapath = Path(__file__).parents[1] / 'data/processed/Validation.h5'

    valid_data = dd.io.load(valid_datapath)
    
    # test the forward and backward pass of the StyleEncoder using sample data
    if False:
        test_StyleEncoder(valid_data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare the dataset 
    train_dataset = StyleEncoderDataset(train_datapath, config['batch_param']['n'], config['batch_param']['m'], dataload_mode='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)

    valid_dataset = StyleEncoderDataset(valid_datapath, config['batch_param']['n'], config['batch_param']['m'], dataload_mode='valid')
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16*8, shuffle=True, drop_last=True)

    model = StyleEncoder()
    model.to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = None
    print('Start training the model on device: {}'.format(device))
    model.train_model(train_dataloader,
                    valid_dataloader,
                    config['batch_param']['n'], 
                    config['batch_param']['m'],
                    optimizer,
                    device,
                    lr_scheduler=scheduler)