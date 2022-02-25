

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

class AnimeDatasetBuilder:

    CACHE_PATH = "/cache/anime_data.npy"

    @staticmethod
    def create_dataset(dataset_path: str, seed = 999, batch_size=64):

        animes_file_path = os.path.join(dataset_path, 'anime_processed.csv')
        rating_file_path = os.path.join(dataset_path, 'ratings_processed.csv')

        animes_df = pd.read_csv(animes_file_path)
        animes_df = animes_df.where(pd.notnull(animes_df), None)

        ratings_df = pd.read_csv(rating_file_path)

        # train_df, test_df = train_test_split(ratings_df, test_size=0.1)
        anime_id_to_attributes = animes_df.set_index('anime_id').to_dict(orient='index')

        if not os.path.isfile(AnimeDatasetBuilder.CACHE_PATH):

            data_matrix = np.ndarray(shape=(len(ratings_df), 11), dtype=np.int)
            ys = np.ndarray(shape=(len(ratings_df)), dtype=np.float)

            for i, row in tqdm(ratings_df.iterrows(), total = len(ratings_df)):

                user_id = row['user_id']
                anime_id = row['anime_id']
                rating = row['rating']
                anime_attributes = anime_id_to_attributes[anime_id]

                data_matrix[i] = user_id, anime_id, anime_attributes['type'], \
                   anime_attributes['episodes'], anime_attributes['rating'], anime_attributes['members'], anime_attributes[
                       'gener1'], \
                   anime_attributes['gener2'], anime_attributes['gener3'], anime_attributes['gener4'], anime_attributes['num_of_genres']

                ys[i] = rating

            with open(AnimeDatasetBuilder.CACHE_PATH, 'wb') as f:
                np.save(f, data_matrix)
                np.save(f, ys)
        else:
            with open(AnimeDatasetBuilder.CACHE_PATH, 'rb') as f:
                data_matrix = np.load(f)
                ys = np.load(f)

        X_train, X_test, y_train, y_test = train_test_split(data_matrix, ys, test_size=0.1, random_state=seed)

        train_tensor_data = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_tensor_data = Data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=True, batch_size=batch_size)

        test_loader = DataLoader(
            dataset=test_tensor_data, shuffle=True, batch_size=batch_size)

        dims = [
                np.max(ratings_df['user_id']) +1,
                np.max(ratings_df['anime_id'])+1,
                np.max(animes_df['type'])+1,
                np.max(animes_df['episodes'])+1,
                np.max(animes_df['rating'])+1,
                np.max(animes_df['members'])+1,
                np.max(animes_df['gener1'])+1,
                np.max(animes_df['gener2'])+1,
                np.max(animes_df['gener3'])+1,
                np.max(animes_df['gener4']) +1,
               np.max(animes_df['num_of_genres']) + 1
                ]

        return RSDataset(dims, train_loader, test_loader)

    @staticmethod
    def get_matrix_dataset(dataset_path: str, seed=999):

        animes_file_path = os.path.join(dataset_path, 'anime_processed.csv')
        rating_file_path = os.path.join(dataset_path, 'ratings_processed.csv')

        animes_df = pd.read_csv(animes_file_path)
        animes_df = animes_df.where(pd.notnull(animes_df), None)

        ratings_df = pd.read_csv(rating_file_path)

        anime_id_to_attributes = animes_df.set_index('anime_id').to_dict(orient='index')

        if not os.path.isfile(AnimeDatasetBuilder.CACHE_PATH):

            data_matrix = np.ndarray(shape=(len(ratings_df), 11), dtype=np.int)
            ys = np.ndarray(shape=(len(ratings_df)), dtype=np.float)

            for i, row in tqdm(ratings_df.iterrows(), total=len(ratings_df)):
                user_id = row['user_id']
                anime_id = row['anime_id']
                rating = row['rating']
                anime_attributes = anime_id_to_attributes[anime_id]

                data_matrix[i] = user_id, anime_id, anime_attributes['type'], \
                                     anime_attributes['episodes'], anime_attributes['rating'], anime_attributes[
                                         'members'], anime_attributes[
                                         'gener1'], \
                                     anime_attributes['gener2'], anime_attributes['gener3'], anime_attributes['gener4'], \
                                     anime_attributes['num_of_genres']

                ys[i] = rating

            with open(AnimeDatasetBuilder.CACHE_PATH, 'wb') as f:
                np.save(f, data_matrix)
                np.save(f, ys)
        else:
            with open(AnimeDatasetBuilder.CACHE_PATH, 'rb') as f:
                data_matrix = np.load(f)
                ys = np.load(f)

        X_train, X_test, y_train, y_test = train_test_split(data_matrix, ys, test_size=0.1, random_state=seed)
        return X_train, X_test, y_train, y_test








