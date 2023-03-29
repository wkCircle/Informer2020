from pathlib import Path 
import pandas as pd 
import numpy as np 

def read_aviation(root_path: str, data_path: str, mode='single-emb'): 
    assert mode in ['single-emb', 'multi-emb'], "Value of mode argument is not allowed."
    df = Path(root_path) / data_path
    df = pd.read_parquet(df).fillna(np.nan) 
    df = df.pivot(index="date", columns=["oper", "actype", "region"], values=["SUM_cycles_act", "SUM_ophrs_act"]) # values order matters! (StandardScaler.inverse_transform)
    df.columns = df.columns.rename({None:'features'})
    df = df.T.reorder_levels(order=['oper', 'actype', 'region', 'features']).T
    if mode == 'single-emb': 
        df = df.filter(regex="SUM_ophrs_act")
    hiercols = df.columns # keep the original hierarchical columns for return 
    df.columns = df.columns.map(".".join) # flatten hierarchical columns 
    return df, hiercols