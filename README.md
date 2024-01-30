# 概要

## 環境構築
conda create -n recommend python=3.7.3
conda activate
pip install -r requirements.txt

pip install ipykernel 
python -m ipykernel install --user --name recommend --display-name "recommend(Python3.7.3)"


## 実行
```sh
python run.py

```