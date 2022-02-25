import os.path

import torch
import tqdm
from typing import Optional
from sklearn.metrics import mean_absolute_error
from rs.datasets.rsdataset import RSDataset
from rs.models.xdfm import ExtremeDeepFactorizationMachineModel
from rs.datasets.anime import AnimeDatasetBuilder
import logging
import torch.nn as nn
import xgboost


xgboost_model_checkpoint = '/rs/RSproject/rs/chkpt/xgbbost_anime.json'
#v1
def get_model(dataset: RSDataset):
    fields_dimension = dataset.fields_dimension
    return ExtremeDeepFactorizationMachineModel(
        fields_dimension, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)

def get_baseline_model():

    xgboost_model = xgboost.XGBRegressor(n_estimators=250, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8,
                                  early_stopping_rounds=20)

    xgboost_model.load_model(xgboost_model_checkpoint)

    return xgboost_model

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


def eval(model, data_loader, device):

    model.eval()
    targets, predicts = list(), list()

    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields = fields.to(device)
            y = model(fields)
            targets.extend(target.detach().cpu().numpy().tolist())
            predicts.extend(y.detach().cpu().numpy().tolist())

    return mean_absolute_error(targets, predicts)

def get_loss_function(loss_function: Optional[str]):

    if loss_function is None or loss_function =='MSE':
        return nn.MSELoss()
    elif loss_function == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError('unkwon loss {}'.format(loss_function))

def main(dataset_path, epoch,
         learning_rate,
         batch_size,
         weight_decay,
         save_dir, loss_function: str):

    device = 'cuda:0' if torch.cuda.is_available()  else 'cpu'
    logging.info("device: {}".format(device))

    dataset = AnimeDatasetBuilder.create_dataset(dataset_path,
                                                 batch_size= batch_size)

    train_data_loader = dataset.train_data_loader
    test_data_loader = dataset.test_data_loader
    model = get_model(dataset).to(device)
    criterion = get_loss_function(loss_function).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    eval_metrics = []
    best_eval_error = 100000

    for epoch_i in range(epoch):
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
            }, os.path.join(save_dir, 'anime_model_checkpoint_v2.pt'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='rs/datasets/anime')
    parser.add_argument('--model_name', default='afi')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--loss_function', type=str, default='MSE')
    parser.add_argument('--save_dir', default='rs/RSproject/rs/chkpt')

    args = parser.parse_args()
    main(
        args.dataset_path,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.save_dir, args.loss_function)