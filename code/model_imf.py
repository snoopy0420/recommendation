import os
from typing import Optional
import numpy as np
import pandas as pd

from model import Model
from util import Util, Metric

import implicit
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix

class Model_iALS(Model):

    def train(self, train_df):
        """
        Args:
            train_df: pd.DataFrame
        """
        # パラメータ
        params = dict(self.params)
        ## 因子数
        factors = params.pop("factors", 10)
        ## エポック数
        n_epochs = params.pop("n_epochs", 50)
        ## alpha
        alpha = params.pop("alpha", 1.0)

        # ユーザーとアイテムのリストを作成
        users = sorted(train_df["user_id"].unique())
        movies = sorted(train_df["movie_id"].unique())

        ## 行列分解用に行列を作成する
        user_id2index = dict(zip(users, range(len(users))))
        movie_id2index = dict(zip(movies, range(len(movies))))
        index_2user_id = dict(zip(range(len(users)), users))
        index_2movie_id = dict(zip(range(len(movies)), movies))

        movielens_matrix = lil_matrix((len(users), len(movies)))
        for i, row in train_df.iterrows():
            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            movielens_matrix[user_index, movie_index] = 1.0 * alpha

        ## csr matrixに変換
        movielens_matrix = csr_matrix(movielens_matrix)

        # モデルの初期化
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, iterations=n_epochs, calculate_training_loss=True, random_state=1
        )

        # 学習
        self.model.fit(movielens_matrix)

        # ユーザーごとに推薦を行い、結果を保存
        self.result = pd.DataFrame()
        for user_id in users:
            user_index = user_id2index[user_id]
            movie_indexes, scores = self.model.recommend(
                userid=user_index,
                user_items=movielens_matrix[user_index],
                N=len(movies)
            )
            movie_ids = [index_2movie_id[movie_index] for movie_index in movie_indexes]
            df = pd.DataFrame({"user_id": [user_id] * len(movie_ids), "item_id": movie_ids, "score": scores})
            self.result = pd.concat([self.result, df])
        # スコア順にソート
        self.result.sort_values(["user_id", "score"], ascending=[True,False])
        
    def predict(self, user_id, k=10):
        """
        Args:
            user_id(int): レコメンドを出力するuser_id
            k(int): レコメンドするアイテム数
        Return:
            レコメンドアイテムのリスト
        """
        # todo:user_idがない場合の処理
        # user_idに紐づくコンテンツを抽出
        list_recomendations = self.result[self.result["user_id"]==user_id]["item_id"].values.tolist()
        # スコアの上位k件に絞り込み
        list_recomendations = list_recomendations[:k]

        return list_recomendations
    
    def recommend(self, list_user_id, k=10):
        """
        Args:
            list_user_id(list): レコメンドを出力するuser_idのリスト
            k(int): レコメンドするアイテム数
        Return:
            各ユーザーに対するレコメンドアイテムのリスト(pd.Dataframe)
        """
        self.dict_user2content = {user_id: self.predict(user_id) for user_id in list_user_id}
        df_recomendations = pd.DataFrame()
        df_recomendations["user_id"] = self.dict_user2content.keys()
        df_recomendations["recommends"] = self.dict_user2content.values()

        return df_recomendations


    