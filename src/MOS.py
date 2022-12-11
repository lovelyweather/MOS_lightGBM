from . import download, preprocess, train

import numpy as np
import glob,os
from datetime import datetime,timedelta
from typing import Dict, List
from functools import partial
from dateutil.parser import parse
import multiprocessing
from multiprocessing import Pool
import cfg
import pickle
from dask_ml.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV 
from dask_ml.metrics import mean_squared_error  
import dask.dataframe as dd
from dask.distributed import Client

'''
需要改存储地址地址、下载时间、需要考虑的varname（暂时不改），会自动处理成cvs。
'''
# download data
save_dir = '/d/work/data'

download.download_gfs_batch(datetime(2022,10,1,6), datetime(2022,10,1,6))
download.download_era5(start="20221001", end="20221003")

# prepare csv
gfs_data_dir: str = "/home/ubuntu/data/gfs/0p25"
era5_data_dir: str = "/home/ubuntu/data/era5/data/"
start_time_list: List[datetime] = [parse(a.split("0p")[-1][3:]) for a in glob.glob(os.path.join(gfs_data_dir, "20??", "??", "??", "??"))]

os.makedirs(cfg.csv_save_dir, exist_ok=True)

for i in glob.glob(os.path.join(gfs_data_dir, "20??", "??", "??", "??", "*.idx")):
    os.system(f"rm {i}")

for i in glob.glob(os.path.join(era5_data_dir, "20??", "??", "*.idx")):
     os.system(f"rm {i}")

varname = ["t2m", "r2", "d2m", "u10", "v10", "mslet" ]
short_name = ["2t", "2r", "2d", "10u", "10v", "mslet"] # mslet: mean sea-level pressure
var_short_dict_gfs = dict(zip(varname, short_name))

varname = ["t2m", "r2", "d2m", "u10", "v10", "msl" ]
short_name = ["2t", "2r", "2d", "10u", "10v", "msl"]
var_short_dict_era5 = dict(zip(varname, short_name))

save_func = partial(
    preprocess.prepare_csv,
    gfs_varname_dict=var_short_dict_gfs,
    era5_varname_dict=var_short_dict_era5,
    gfs_data_dir=gfs_data_dir,
    era5_data_dir=era5_data_dir,
    save_dir=cfg.csv_save_dir) #partial:固定函数的一些参数

job_list = []
for dtime in start_time_list:
    for ifcst in range(3, 24+1, 3):
        job_list.append((dtime, ifcst))
        #[(datetime.datetime(2022, 9, 1, 6, 0), 3),  (datetime.datetime(2022, 9, 1, 6, 0), 6), ...]

njobs: int = multiprocessing.cpu_count()

with Pool(njobs) as p:
    p.starmap(save_func, job_list)

# split dataset, train and tuning

X, y = train.process_X_y(var_MOS = 't2m')
X_train, X_test, y_train, y_test = train_test_split(X, y,
     random_state=0, test_size = 0.2)

# train, save model
train.gen_model(X_train, y_train, var_MOS = 't2m', search_cv = None, is_save_model = True)

# predict!
for i, i_var_era5 in enumerate(["t2m"]):
    # 读取预测模型
    with open(os.path.join(cfg.model_dir, f"{i_var_era5}_lgb.pkl"), "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred))
    print(" RMSE: %f" % (rmse_lgb ))

