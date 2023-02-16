from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from pathlib import Path 
import re, os, sys 
import pickle 

# utils import 
project_dir = Path(re.search(".*Informer2020", os.getcwd())[0])
sys.path.append(project_dir.as_posix())
from utils.read_aviation import read_aviation
# out-of-scope package import
fh_project_dir = Path.home() / "Bitbucket/fh_forecast_model"
package_path = fh_project_dir / "src/core/io"
sys.path.append(package_path.as_posix())
from writers import save_train_test

# ========== Convert output ==========
def glob_re(pattern: str, iterable: Iterable):
    """get matched items in iterable with pattern and return a generator."""
    return filter(re.compile(pattern).search, iterable)

name = "test_forecast.parquet.gzip"
root_dir = project_dir / "results"
regex = "informer_Aviation_\d{2}_*"
dirs = glob_re(regex, os.listdir(root_dir))
for fstring in dirs:
    path = root_dir / fstring
    if name in os.listdir(path):
        continue 
    # read pred, meta, and transformed traindf
    values = np.load(path / f"real_prediction.npy").squeeze()
    with (path / f"train_meta.pkl").open('rb') as fr: 
        meta = pickle.load(fr) 
    traindf, hiercols = read_aviation(project_dir / meta['root_path'], meta['data_path'], mode=meta['mode'])
    assert values.shape[-1] == len(hiercols)
    # recover test pred format 
    timestamps = pd.date_range(start=traindf.index.max(), periods=len(values)+1, freq='MS')[1:]
    tes_pred = pd.DataFrame(values, index=timestamps, columns=hiercols).T
    tes_pred = tes_pred.melt(
        var_name='date', value_name='SUM_ophrs_act', 
        ignore_index=False
    ).reset_index()
    # save to fstring dir 
    save_train_test(
        parent=path, train_data=None, test_data=tes_pred, 
        name_suffix="forecast", index=False
    )
    