


from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ModuleList

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias







class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims_with_embedding_dims: Tuple):
        super().__init__()
        self.embeddings =  ModuleList([torch.nn.Embedding(field_dim, embed_dim) for (field_dim, embed_dim) in field_dims_with_embedding_dims])

        [torch.nn.init.xavier_uniform_(embedding.weight.data) for embedding in self.embeddings]

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        num_of_features = x.shape[1]

        all_embeddings_results = []

        for feature in range(num_of_features):
            all_embeddings_results.append(self.embeddings[feature](x[:, feature]))

        xx = [x.unsqueeze(1) for x in all_embeddings_results]

        return torch.concat(xx, dim=2)




class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)



class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class ImprovedExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.
    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims_with_embedding_dims, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims_with_embedding_dims)
        self.embed_output_dim = np.sum([x[1] for x in field_dims_with_embedding_dims])
        self.cin = CompressedInteractionNetwork(len(field_dims_with_embedding_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear([x[0] for x in field_dims_with_embedding_dims])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1)


import os.path

import torch
import tqdm
from typing import Optional
from sklearn.metrics import mean_absolute_error
from rs.datasets.rsdataset import RSDataset
from rs.models.improvedxdfm import ImprovedExtremeDeepFactorizationMachineModel
from rs.datasets.anime import AnimeDatasetBuilder
import logging
import torch.nn as nn
import xgboost


def get_model(layer_dimension=16, num_of_cim_layers=2,
              mlp_layer_size=2):

    fields_dimension = [(73517,16), (34476,16), (7,16), (1819,16), (20,5), (1013918,8),
                        (44,4), (44,4), (44,4), (44,4), (14,4)]

    cross_layer_sizes = [layer_dimension for _ in range(num_of_cim_layers)]
    mlp_layer_sizes = [layer_dimension for _ in range(mlp_layer_size)]

    return ImprovedExtremeDeepFactorizationMachineModel(
        fields_dimension, cross_layer_sizes=cross_layer_sizes,
        split_half=False, mlp_dims=mlp_layer_sizes,
        dropout=0.2)


dataset_path = '/home/gabib/workspace/rs/datasets/anime'
model_checkpoint_dir = 'models/anime/xdfm'

def get_loss_function(loss_function: Optional[str]):

    return nn.MSELoss()



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logging.info("device: {}".format(device))

batch_size = 32
dataset = AnimeDatasetBuilder.create_dataset(dataset_path,
                                             batch_size=batch_size)

train_data_loader = dataset.train_data_loader
test_data_loader = dataset.test_data_loader
model = get_model(layer_dimension=8,
                  num_of_cim_layers=2, mlp_layer_size=2).to(device)

loss_function = get_loss_function(None)
criterion = get_loss_function(loss_function).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

eval_metrics = []
best_eval_error = 100000

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    epoc_loss = 0.0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)

    total_rows = 0.0
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        total_rows += target.shape[0]
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        epoc_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

    print('EPOC loss {}'.format(epoc_loss/len(data_loader)))

for epoch_i in range(10):

    print('start epoc {}'.format(epoch_i))

    train(model, optimizer, train_data_loader, criterion, device)
    test_mean_abs_error = eval(model, test_data_loader, device)
    train_mean_abs_error = eval(model, train_data_loader, device)

    print('MEAN ABS train: {} validation: {}'.format(train_mean_abs_error, test_mean_abs_error))
    eval_metrics.append((train_mean_abs_error, test_mean_abs_error))

    if best_eval_error > test_mean_abs_error:
        best_eval_error = test_mean_abs_error
        print('save model checkpoint')
        torch.save({
            'epoch': epoch_i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join("dsa", 'anime_model_checkpoint_{}.pt'.format(test_mean_abs_error)))
