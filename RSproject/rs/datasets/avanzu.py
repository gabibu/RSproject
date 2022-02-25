
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()
import torch.utils.data as Data
import torch
import os
from rs.datasets.rsdataset import RSDataset

class DatasetLoader(RSDataset):

    def __init__(self, ratings_df, animes_df):
        self._ratings_df = ratings_df

        self._anime_id_to_attributes = animes_df.set_index('anime_id').to_dict(orient='index')


    def __len__(self):
        return len(self._ratings_df)

    def __getitem__(self, idx):
        user_rating_row = self._ratings_df.iloc[idx]
        user_id = user_rating_row['user_id']
        anime_id = user_rating_row['anime_id']
        rating = user_rating_row['rating']

        anime_attributes = self._anime_id_to_attributes[anime_id]


        return user_id, anime_id, anime_attributes['type'],\
               anime_attributes['episodes'], anime_attributes['rating'], anime_attributes['members'], anime_attributes['gener1'], \
               anime_attributes['gener2'], anime_attributes['gener3'], anime_attributes['gener4'], rating

class AvanzuDatasetBuilder:


    CASH_PATH = 'cache/rs/RSproject/cache/avanzu.npy'

    FEATURES_COLUMNS_NAME = ['time', 'C1', 'banner_pos', 'site_id', 'site_domain',
                             'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                             'device_ip', 'device_model', 'device_type', 'device_conn_type',
                             'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

    @staticmethod
    def create_dataset(dataset_path: str, seed = 999, batch_size=64):

        print('read')
        df = pd.read_csv(os.path.join(dataset_path, 'train_processed.csv'))
        print('loaded')

        #df = df.where(pd.notnull(df), None)


        if not os.path.isfile(AvanzuDatasetBuilder.CASH_PATH):

            data_matrix = np.ndarray(shape=(len(df), 22), dtype=np.int)
            ys = np.ndarray(shape=(len(df)), dtype=np.float)

            for i, row in tqdm(df.iterrows(), total = len(df)):

                data_matrix[i] = [row[col] for col in AvanzuDatasetBuilder.FEATURES_COLUMNS_NAME]

                ys[i] = row['click']

            with open(AvanzuDatasetBuilder.CASH_PATH, 'wb') as f:
                np.save(f, data_matrix)
                np.save(f, ys)
        else:
            with open(AvanzuDatasetBuilder.CASH_PATH, 'rb') as f:
                data_matrix = np.load(f)
                ys = np.load(f)

        X_train, X_test, y_train, y_test = train_test_split(data_matrix, ys, test_size=0.1, random_state=seed)

        train_tensor_data = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_tensor_data = Data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=True, batch_size=batch_size)

        test_loader = DataLoader(
            dataset=test_tensor_data, shuffle=True, batch_size=batch_size)

        dims = [np.max(df[col]) + 1 for col in AvanzuDatasetBuilder.FEATURES_COLUMNS_NAME]

        return RSDataset(dims, train_loader, test_loader)

    @staticmethod
    def create_matrix_dataset(dataset_path: str, seed=999):

        print('read')
        df = pd.read_csv(os.path.join(dataset_path, 'train_processed.csv'))
        print('loaded')

        # df = df.where(pd.notnull(df), None)

        if not os.path.isfile(AvanzuDatasetBuilder.CASH_PATH):

            data_matrix = np.ndarray(shape=(len(df), 22), dtype=np.int)
            ys = np.ndarray(shape=(len(df)), dtype=np.float)

            for i, row in tqdm(df.iterrows(), total=len(df)):
                data_matrix[i] = [row[col] for col in AvanzuDatasetBuilder.FEATURES_COLUMNS_NAME]

                ys[i] = row['click']

            with open(AvanzuDatasetBuilder.CASH_PATH, 'wb') as f:
                np.save(f, data_matrix)
                np.save(f, ys)
        else:
            with open(AvanzuDatasetBuilder.CASH_PATH, 'rb') as f:
                data_matrix = np.load(f)
                ys = np.load(f)

        X_train, X_test, y_train, y_test = train_test_split(data_matrix, ys, test_size=0.1, random_state=seed)

        return X_train, X_test, y_train, y_test

