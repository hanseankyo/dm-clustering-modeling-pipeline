from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def eGFR_cal(cr: float, age: float, sex: int) -> float:
    if sex == 1:
        k = 0.7
        alpha = -0.241
        last = 1.012
    else:
        k = 0.9
        alpha = -0.302
        last = 1.0

    egfr = 142 * ((min(cr / k, 1)) ** alpha) * ((max(cr / k, 1)) ** (-1.2)) * ((0.9938) ** age) * last
    return float(np.round(egfr, 0))


def mk_eGFR_data(clinical_only_raw):
    egfr_data = []
    for sample in range(len(clinical_only_raw)):
        egfr_data.append(
            eGFR_cal(
                clinical_only_raw.loc[sample, "Cr"],
                clinical_only_raw.loc[sample, "age"],
                clinical_only_raw.loc[sample, "F"],
            )
        )
    return egfr_data


def mkMBP(cli):
    cli["MBP"] = (2 * cli["dia"] + cli["sys"]) / 3
    cli = cli.drop(columns=["dia", "sys"])
    return cli


def _to_1d_bin(x):
    if "torch" in str(type(x)):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    if x.dtype not in (np.int64, np.int32, np.int8):
        x = (x > 0.5).astype(int)
    return x


def check_correct(predict, y):
    y = _to_1d_bin(y)
    predict = _to_1d_bin(predict)

    if y.size == 0 or predict.size == 0 or y.size != predict.size:
        return np.nan, np.nan, np.nan

    tp = np.sum((predict == 1) & (y == 1))
    tn = np.sum((predict == 0) & (y == 0))
    fp = np.sum((predict == 1) & (y == 0))
    fn = np.sum((predict == 0) & (y == 1))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else np.nan
    pos_den = tp + fn
    sen = tp / pos_den if pos_den > 0 else np.nan
    neg_den = tn + fp
    spe = tn / neg_den if neg_den > 0 else np.nan
    return acc, sen, spe


class FCNetwork(nn.Module):
    def __init__(self, feature_num: int):
        super().__init__()
        self.pre_part = nn.Sequential(
            nn.Linear(feature_num, feature_num),
            nn.BatchNorm1d(feature_num),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(feature_num, feature_num),
            nn.BatchNorm1d(feature_num),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(feature_num, feature_num),
            nn.BatchNorm1d(feature_num),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(feature_num, feature_num),
            nn.BatchNorm1d(feature_num),
            nn.ELU(),
            nn.Dropout(),
        )
        self.fc4 = nn.Linear(feature_num, 1)

    def forward(self, x):
        x = self.pre_part(x)
        return self.fc4(x)
