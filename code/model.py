import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Optional


class Model(metaclass=ABCMeta):
    """
    model_xxのスーパークラス
    abcモジュールにより抽象メソッドを定義
    """

    def __init__(self, run_fold_name: str, params: dict) -> None:
        """コンストラクタ
        run_fold_name: runの名前とfoldの番号を組み合わせた名前
        params: ハイパーパラメータ
        """
        self.run_fold_name = run_fold_name
        self.params = params
        self.model = None

    @abstractmethod
    def train(self,
              tr_x: pd.DataFrame,
              tr_y: Optional[pd.DataFrame] = None,
              va_x: Optional[pd.DataFrame] = None,
              va_y: Optional[pd.Series] = None
            ) -> None:
        """モデルの学習を行い、学習済のモデルを保存する
        Args:
            tr_x: 学習データの特徴量
            tr_y: 学習データの目的変数
            va_x: バリデーションデータの特徴量
            va_y: バリデーションデータの目的変数
        バリデーションデータはlightgbmなどの学習時にバリデーションデータを使用するモデルの場合に使用(early stopping)
        """
        pass

    def predict(self, te_x: pd.DataFrame) -> list:
        """学習済のモデルでの予測値を返す
        :param te_x: バリデーションデータやテストデータの特徴量
        :return: 予測値(予測確率）
        """
        pass

    @abstractmethod
    def recommend(self, list_user_id: list) -> pd.DataFrame:
        """学習済のモデルでの予測値を返す
        :param te_x: バリデーションデータやテストデータの特徴量
        :return: 予測値(予測確率）
        """
        pass

    def save_model(self) -> None:
        """モデルの保存を行う"""
        pass

    def load_model(self) -> None:
        """モデルの読み込みを行う"""
        pass
