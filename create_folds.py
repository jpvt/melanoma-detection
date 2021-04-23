import os
import pandas as pd
from sklearn import model_selection


DATAPATH = "data/"

if __name__ == "__main__":

    df = pd.read_csv("data/train.csv")
    df['kfold'] = -1
    # new and shuffled dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    skf = model_selection.StratifiedKFold(n_splits=10)

    for fold_, (_, _) in enumerate(skf.split(X=df, y=y)):

        df.loc[:, "kfold"] = fold_

    df.to_csv("data/train_folds.csv", index=False)
