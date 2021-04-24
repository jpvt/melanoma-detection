import os
import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":

    df = pd.read_csv("../data_storage/train.csv")
    df['kfold'] = -1
    # new and shuffled dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    skf = model_selection.StratifiedKFold(n_splits=10)

    for fold_, (_, v_) in enumerate(skf.split(X=df, y=y)):

        df.loc[v_, "kfold"] = fold_

    df.to_csv("../data_storage/train_folds.csv", index=False)
