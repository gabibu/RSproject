
import torch
import tqdm
from rs.datasets.anime import AnimeDatasetBuilder
from rs.datasets.avanzu import AvanzuDatasetBuilder
import pandas as pd

class PredictionsUtils:

    @staticmethod
    def get_xdfm_users_anime_predictions(model, dataset_path: str, device):

        model.eval()
        model.to(device)
        data_loader = AnimeDatasetBuilder.create_dataset(dataset_path).test_data_loader

        targets, predicts, users, animes = list(), list(), list(), list()

        with torch.no_grad():
            for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                    users.extend(fields[:, 0].numpy().tolist())
                    animes.extend(fields[:, 1].numpy().tolist())

                    fields = fields.to(device)
                    y = model(fields)
                    targets.extend(target.detach().cpu().numpy().tolist())
                    predicts.extend(y.detach().cpu().numpy().tolist())

            res =  list(zip(users, animes, targets, predicts))
            return pd.DataFrame(res, columns=['user_id', 'anime_id', 'target', 'prediction'])

    @staticmethod
    def get_xgboost_users_anime_predictions(model, dataset_path):

        _, X_test, _, y_test = AnimeDatasetBuilder.get_matrix_dataset(dataset_path)

        predictions = model.predict(X_test)

        users_animes_data = []
        for i in range(predictions.shape[0]):
            users_animes_data.append((X_test[i][0], X_test[i][1], y_test[i], predictions[i]))

        return pd.DataFrame(users_animes_data, columns=['user_id', 'anime_id', 'target', 'prediction'])

    @staticmethod
    def get_xdfm_avanzu_predictions(model, dataset_path: str, device):

        model.eval()
        model.to(device)
        data_loader = AvanzuDatasetBuilder.create_dataset(dataset_path).test_data_loader

        targets, predicts, users, animes = list(), list(), list(), list()

        sites_ids = []
        with torch.no_grad():
            for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):

                sites_ids.extend(fields[:, 3].numpy().tolist())
                fields = fields.to(device)
                y = model(fields)
                targets.extend(target.detach().cpu().numpy().tolist())
                predicts.extend(y.detach().cpu().numpy().tolist())

            res = list(zip(targets, predicts, sites_ids))
            return pd.DataFrame(res, columns=['target', 'prediction', 'site_id'])

    @staticmethod
    def get_xgboost_avanzu_predictions(model, dataset_path):

        _, X_test, _, y_test = AvanzuDatasetBuilder.create_matrix_dataset(dataset_path)

        predictions = model.predict(X_test)

        labels_prediction_data = []
        for i in range(predictions.shape[0]):
            labels_prediction_data.append((y_test[i], predictions[i], X_test[i][3]))

        return pd.DataFrame(labels_prediction_data, columns=['target', 'prediction', 'site_id'])






