from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
np.random.seed(42)
import deepdish as dd
from pathlib import Path
import yaml
import umap

class StyleEncoder(nn.Module):
    
    def __init__(self,
                 load_model=False,
                 epoch=1,
                 device=torch.device('cpu')):
        
        super(StyleEncoder, self).__init__()
        
        self.config = yaml.load(open('config.yml', 'r'), Loader=yaml.SafeLoader) 

        self.device = device
        self.hidden_size = self.config['hidden_size']
        
        self.final_embedding_size = self.config['embedding_size']

        self.lstm = nn.LSTM(input_size=self.config['input_size'],
                            hidden_size=self.config['hidden_size'], 
                            num_layers=self.config['num_layers'], 
                            batch_first=True)
        self.linear = nn.Linear(in_features=self.config['hidden_size'], 
                                out_features=self.config['embedding_size'])

       
        self.load_model = load_model
        self.epoch = epoch
        self.optimizer = None
        # self.config = dict(self.config)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, x, hidden_init=None):
        out, (hidden, cell) = self.lstm(x, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds = F.relu(self.linear(hidden[-1]))
        
        embeds = embeds.view(-1, self.config['embedding_size'])
        embeds = embeds.clone() / (torch.norm(embeds, dim=1, keepdim=True) + 1e-5)

        return embeds
              
    def similarity_matrix(self, embeds):
        """ Computes the similarity matrix for Generalized-End-To-End-Loss 
        embeds : Embedding tensor of shape (subjects_per_batch, samples_per_subject, embedding_size)
        function used from:
        https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/95adc699c1deb637f485e85a5107d40da0ad94fc/encoder/model.py#L33
        """        
        subjects_per_batch, samples_per_subject = embeds.shape[:2]

        # centroid inclusive (eq. 1)
        # (Cj / |Cj|) gives the unit vector which is later used for finding cosine similarity 
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5) 
        
        
        # centroid exclusive (eq. 8)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (samples_per_subject - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(subjects_per_batch, samples_per_subject,
                                 subjects_per_batch).to(self.device)
        mask_matrix = 1 - np.eye(subjects_per_batch, dtype=np.int)
        for j in range(subjects_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias

        return sim_matrix
        
    def total_loss(self, embeds, labels):
        """
        Computes the softmax loss according the section 2.1 of Generalized End-To-End loss.
        
        embeds: the embeddings as a tensor of shape (subjects_per_batch, 
        samples_per_subject, embedding_size)
        """
        subjects_per_batch, samples_per_subject = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((subjects_per_batch * samples_per_subject, 
                                         subjects_per_batch))
        ground_truth = np.repeat(np.arange(subjects_per_batch), samples_per_subject)
        target = torch.from_numpy(ground_truth).long().to(self.device)
        GE2E_loss = self.loss_fn(sim_matrix, target)

        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, subjects_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return GE2E_loss, eer
    
    def accuracy(self, output, labels):
        predictions = torch.argmax(output, dim=1)   
        correct = torch.sum(torch.eq(predictions, labels))

        accuracy = 100 * correct / labels.shape[0]
        
        return accuracy.detach().cpu().numpy()
      
    def load_model_from_dict(self, checkpoint):
        """ Load the model from the checkpoint by filtering out the unnecessary parameters"""
        model_dict = self.state_dict()
        # filter out unnecessary keys in the imported model
        pretrained_dict = {k:v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
        # overwrite the entries in the existing state dictionary
        model_dict.update(pretrained_dict)
        # load the new state dict
        self.load_state_dict(model_dict)     
        if "use_speaker_embeds" in checkpoint.keys():
            self.use_speaker_embeds = checkpoint['use_speaker_embeds']
                
    def train_model(self,
                   train_dataloader,
                   valid_dataloader,
                   subjects_per_batch, 
                   samples_per_subject,
                   optimizer,
                   device,
                   lr_scheduler=None,
                   load_model=False,
                   checkpoint=None):

        self.device = device

        model_log_dir = os.path.join(
            self.config['model_save_dir'], '{}'.format(self.__class__.__name__))
        run_log_dir = os.path.join(
            self.config['runs_dir'], '{}'.format(self.__class__.__name__))
        
        if not load_model:  
            model_save_dir = os.path.join(os.path.join(
                                      model_log_dir, "run_{}".format(
                                      len(os.listdir(model_log_dir)) if os.path.exists(model_log_dir) else 0))
                                      )
            self.model_save_string = os.path.join(
                    model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')
            
            os.makedirs(model_save_dir, exist_ok=True)
            os.makedirs(self.config['vis_dir'], exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=os.path.join(
                                        run_log_dir, "run_{}".format(
                                        len(os.listdir(run_log_dir)) if os.path.exists(run_log_dir) else 0)))
        else:
            model_save_dir = os.path.join(os.path.join(
                                      model_log_dir, "run_{}".format(len(os.listdir(model_log_dir)) - 1 if os.path.exists(model_log_dir) else 0)))
            self.model_save_string = os.path.join(
                model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')       


        if load_model:
            self.load_model_from_dict(checkpoint=checkpoint)

        for epoch in range(1, self.config['train_epochs']):
            self.epoch = epoch
            # set the model to training mode
            self.train(mode=True)
            loss_vec, eer_vec = [], []
            embeds_vec, label_vec = [], []
            for _, data in enumerate(train_dataloader):

                features = data['features'].squeeze().to(self.device)
                labels = data['labels'].squeeze().to(self.device)

                embeds = self.forward(features)
                loss, eer = self.total_loss(embeds.reshape(subjects_per_batch, samples_per_subject, self.final_embedding_size), labels)

                optimizer.zero_grad()
                loss.backward()
                self.do_gradient_ops()
                optimizer.step()
                                
                loss_vec.append(loss.data.item())
                eer_vec.append(eer)
                embeds_vec.append(embeds.detach().cpu().numpy())
                label_vec.append(labels.cpu().numpy())
                
            eer_vec_valid, loss_vec_valid, embeds_vec_valid, label_vec_valid = [], [], [], []
            for i, data_valid in enumerate(valid_dataloader): 
                features = data_valid['features'].squeeze().to(self.device)
                labels = data_valid['labels'].squeeze().to(self.device)

                valid_embeds = self.forward(features)
                # accuracy = self.accuracy(output, data_valid['labels'].to(self.device))
                loss_valid, eer_valid = self.total_loss(valid_embeds.reshape(subjects_per_batch, samples_per_subject, self.final_embedding_size), labels)
                
                loss_vec_valid.append(loss_valid.data.item())
                eer_vec_valid.append(eer_valid)
                embeds_vec_valid.append(valid_embeds.detach().cpu().numpy())
                label_vec_valid.append(labels.cpu().numpy())

            self.writer.add_scalars('Running loss', {'Training':np.mean(loss_vec),
                                                    'Validation':np.mean(loss_vec_valid)}, epoch)
            self.writer.add_scalars('EER', {'Training':np.mean(eer_vec),
                                                'Validation':np.mean(eer_vec_valid)}, epoch)

            if lr_scheduler:
                if epoch % 50 == 0:
                    lr_scheduler.step()
                
            if epoch % 100 == 0:
                print("Device: {}, Epoch: {}, Loss: {}, EER- train:{}, Validation:{}".format(self.device, epoch, np.mean(loss_vec), np.mean(eer_vec), np.mean(eer_vec_valid)))
                                
                torch.save(
                    {   'epoch': self.epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, self.model_save_string.format(epoch))
            
                reducer = umap.UMAP()
                umap_embeds_train = reducer.fit_transform(np.concatenate(embeds_vec, axis=0))
                umap_embeds_valid = reducer.fit_transform(np.concatenate(embeds_vec_valid, axis=0))

                _, ax = plt.subplots(1,2)
                ax[0].scatter(umap_embeds_train[:, 0], umap_embeds_train[:, 1], c=np.concatenate(label_vec, axis=0), cmap='viridis')
                ax[0].set_title('Training data embeddings')
                ax[1].scatter(umap_embeds_valid[:, 0], umap_embeds_valid[:, 1], c=np.concatenate(label_vec_valid, axis=0), cmap='viridis')
                ax[1].set_title('Validation data embeddings')
                savepath = self.config['vis_dir']
                os.makedirs(savepath, exist_ok=True)

                plt.savefig(os.path.join(savepath, 'epoch_{}.png'.format(self.epoch)))
                            
    def validate_model(self, dataloader, subjects_per_batch, samples_per_subject, checkpoint=None, savefig=True):
        if checkpoint:
            self.load_model_from_dict(checkpoint)
        
        self.eval()
        embeds_vec, label_vec, eer_vec, loss_vec = [], [], [], []
        
        for i, data in enumerate(dataloader):      

            features = data['features'].squeeze().to(self.device)
            labels = data['labels'].squeeze().to(self.device)
                           
            embeds = self.forward(features)
            loss, eer = self.total_loss(embeds.reshape(subjects_per_batch, samples_per_subject, self.final_embedding_size), labels)
                
            # accuracy = self.accuracy(output, data['labels'].to(self.device))
            
            embeds_vec.append(embeds.detach().cpu().numpy())
            label_vec.append(labels.cpu().numpy())
            eer_vec.append(eer)
            loss_vec.append(loss.data.item())

        reducer = umap.UMAP()
        umap_embeds = reducer.fit_transform(np.concatenate(embeds_vec, axis=0))
        
        print("Emotion prediction EER:{}, loss:{}".format(np.mean(eer_vec), np.mean(loss_vec)))
        plt.figure()
        plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=np.concatenate(label_vec, axis=0), cmap='viridis')
        savepath = self.config['vis_dir']
        os.makedirs(savepath, exist_ok=True)

        if savefig:
            plt.savefig(os.path.join(savepath, 'Test_data_{}.png'.format(self.epoch)))
            
# Encoder dataset
class StyleEncoderDataset(data.Dataset):
    """Create Dataset for style encoder"""
    
    def __init__(self, emg_data_path, subjects_per_batch, samples_per_subject, dataload_mode='train'):
        super(StyleEncoderDataset).__init__()
        self.subjects_per_batch  = subjects_per_batch
        self.samples_per_subject = samples_per_subject
        
        self.batchsize = self.subjects_per_batch * self.samples_per_subject
        self.datapath = emg_data_path
        self.mode = dataload_mode
        
        self.features, self.labels = self.import_data()
            
    def import_data(self):
        data = dd.io.load(self.datapath)
        
        return data['features'], data['labels']
    
    def __len__(self):
        if self.mode == 'train':
            data_len = int(np.floor((self.features.shape[0] - self.batchsize) / (self.batchsize)))
        else:
            data_len = self.features.shape[0]
            
        return data_len 
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist() 
        
        if self.mode == 'train':
            # Data has been preprocessed to form batches, retrieve accordingly
            start = self.batchsize * index 
            stop = self.batchsize * (index + 1)
            
            samples = {'features': torch.tensor(self.features[start:stop, :, :], dtype=torch.float32, requires_grad=True), 
                    'labels': torch.tensor(self.labels[start:stop], dtype=torch.long)}
        else:
            samples = {'features': torch.tensor(self.features[index, :, :], dtype=torch.float32, requires_grad=True), 
                    'labels': torch.tensor(self.labels[index], dtype=torch.long)}
            
        
        return samples