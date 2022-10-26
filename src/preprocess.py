import argparse
import configparser
import glob
import multiprocessing
import os
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Pool
from typing import Dict, List
import numba
import numpy as np
import pandas as pd
import xarray as xr
from dateutil.parser import parse
from loguru import logger
import pygrib as pg

def prepare_csv(
    file_time: datetime,
    ifore: int,
    gfs_varname_dict: dict[str,str],  #  ["t2m","r2","d2m","u10","v10","pres"]
    era5_varname_dict: dict[str,str],  # = ["t2m","r2", "u10","v10","pres"],
    gfs_data_dir: str,    
    era5_data_dir: str,    
    save_dir: str) -> None:
    '''
    :param file_time:         datetime, 每次的gfs预测起始时间（UTC）
    :param ifore:             int, 预报的时次
    :param gfs_varname_dict:  List[str], 存储gfs变量的列表
    :param era5_varname_dict: List[str], 存储era5变量的列表
    :param gfs_data_dir:      str, gfs存储路径
    :param era5_data_dir:     str, era5存储路径
    :param save_dir:          str, csv文件存储路径
    '''
    csv_save_path: str = os.path.join(save_dir,
        f"{file_time:%Y}",
        f"{file_time:%m}",
        f"{file_time:%Y%m%d%H%M%S}_{ifore}.csv",
    )
    if os.path.exists(csv_save_path):
        logger.warning(f"{file_time:%Y%m%d%H%M%S}_{ifore}.csv exists")
    else:
        logger.info(f"begin the process to save the training data file: {file_time:%Y%m%d%H%M%S}_{ifore}.csv")

        # get the variables from GFS and ERA5 data
        gfs_dict = get_var_gfs(data_dir = gfs_data_dir, file_time=file_time, ifore=ifore, var_short_dict=gfs_varname_dict)
        ana_time: datetime = file_time + timedelta(hours=ifore)
        era5_dict = get_var_era5(data_dir=era5_data_dir, file_time=ana_time, var_short_dict=era5_varname_dict)

        len_lat, len_lon = 361, 721 
        # 二维经纬度列表
        array_lat = np.zeros((len_lat, len_lon), dtype=np.float32)
        array_lon = np.zeros((len_lat, len_lon), dtype=np.float32)
        for i in range(0, len_lat, 1):  
            for j in range(0, len_lon, 1):
                array_lat[i, j] = 90 - i * 0.25
                array_lon[i, j] = 0.0 + j * 0.25

        array_total = np.zeros((len_lat, len_lon, 0), dtype=np.float32)
        column_list: List = []

        # 数组叠加拼接
        array_total = np.concatenate((array_total, array_lat[:, :, np.newaxis]), axis=2) #np.newaxis的功能是增加新的维度, 放在哪个位置，就会给哪个位置增加维度,这里理解为竖着存放
        array_total = np.concatenate((array_total, array_lon[:, :, np.newaxis]), axis=2)
        column_list.append("lat")
        column_list.append("lon")

        # ifore列表
        array_ifore = np.ones((len_lat, len_lon), dtype=np.float32) * ifore
        array_total = np.concatenate((array_total, array_ifore[:, :, np.newaxis]), axis=2)
        column_list.append("ifore")

        # 叠加gfs9格点和era5数据
        for i_var_era5 in list(era5_varname_dict.keys()):
            array_era5 = era5_dict[i_var_era5].values
            array_total = np.concatenate((array_total, array_era5[:, :, np.newaxis]), axis=2)
            column_list.append(i_var_era5)

        for i_var_gfs in list(gfs_varname_dict.keys()):
            array_gfs = get_nearby(gfs_dict[i_var_gfs].values,1)
            array_total = np.concatenate((array_total, array_gfs), axis=2)
            for i in range(9):
                column_list.append(i_var_gfs + "_" + str(i))
        
        array_total = np.reshape(array_total, (len_lat * len_lon, -1)) #原来是维度是（361，721，n），n个特征，展开为（361*721，n）

        df = pd.DataFrame(array_total, columns=column_list)
        os.makedirs(os.path.join(save_dir, f"{file_time:%Y}", f"{file_time:%m}"), exist_ok=True)
        df.to_csv(csv_save_path)
        logger.success(f"{file_time:%Y%m%d%H%M%S}_{ifore}.csv done")

    return 1

def get_var_gfs(data_dir: str, file_time: datetime, ifore : int, var_short_dict: dict ):
    '''
    usage: substract the variables of interest in GFS data into a dictionary dataset
    
    params var_short_dict:  var_name <-> short_name
        eg:  var_name: ['t2m', 'r2', 'd2m', 'u10', 'v10', 'mslet'] 
            d2m: 2m dewpoint temperature; mslet: mean sea level pressure using eta reduction. 

    return dr: a dictionary made of variables:xr.DataArray
    '''
    filename = os.path.join(data_dir,  
        f"{file_time:%Y}",
        f"{file_time:%m}",
        f"{file_time:%d}",
        f"{file_time:%H}",
        f"{file_time:gfs.t%Hz.pgrb2.0p25.f{str(ifore).zfill(3)}}") #格式化打印方法
    
    if False in [ list(var_short_dict.keys())[i] in ['t2m', 'r2', 'd2m', 'u10', 'v10', 'mslet'] for i in range(0,len(var_short_dict.items())) ]:
        raise ValueError("we haven't considered the varname yet")
    else: 
        dr = {}
        for var_name,short_name in var_short_dict.items():
            dr[var_name] = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={"shortName": short_name})[var_name].sel(latitude=slice(90, 0), longitude=slice(0, 180)) 
            #shortName和varname不一样，还都需要用到，一一对应，所以建立了dict
    return dr  

def get_var_era5(data_dir: str, file_time: datetime, var_short_dict: dict ):
    '''
    usage: substract the variables of interest in era5 data into a dictionary dataset
    
    params var_short_dict:  var_name <-> short_name
        eg:  var_name: ['t2m', 'r2', 'd2m', 'u10', 'v10', 'mslet'] 
            d2m: 2m dewpoint temperature; mslet: mean sea level pressure using eta reduction. 

    return dr: a dictionary made of variables:xr.DataArray     
    '''
    # /home/ubuntu/data/era5/data/
    filename = os.path.join(data_dir,  
        f"{file_time:%Y}",
        f"{file_time:%m}",
        f"{file_time:%Y%m%d}"+"_sl.grib") #formatted printing
    if False in [ list(var_short_dict.keys())[i] in ['t2m', 'r2', 'd2m', 'u10', 'v10', 'msl'] for i in range(0,len(var_short_dict.items())) ]:
        raise ValueError("we haven't considered the varname yet")
    else:
        dr = {}
        for var_name,short_name in var_short_dict.items():
            if var_name == 'r2':
                # using temperature and dewpoint to calculate relative humidity
                t2m = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={"shortName": "2t"})["t2m"].sel(time=file_time)
                d2m = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={"shortName": "2d"})["d2m"].sel(time=file_time)
                gc = 461.5  # [j/{kg-k}]   gas constant water vapor
                gc = gc / (1000.0 * 4.186)  # [cal/{g-k}]  change units
                # lhv=latent heat vap
                lhv = 597.3 - 0.57 * (t2m - 273.15)  # dutton top p273 [empirical]
                rh = np.exp((lhv / gc) * (1.0 / t2m - 1.0 / d2m))  # type: ignore
                dr[var_name] = rh * 100.
            else:
                dr[var_name] = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={"shortName": short_name})[var_name].sel(time=file_time)

    return dr  

@numba.jit
def get_nearby(a: np.ndarray, radius: int) -> np.ndarray:
    """
    usage: given an array of shape (x,y), get the neighbour points (radius)
        for every grid point, return an array of n*y*radius
    author: haiqin chen

    :param a: the input array
    :type a: np.ndarray
    :return b: the output array
    :type b: np.ndarray
    """
    b = np.zeros(shape=(a.shape[0],a.shape[1],(2*radius+1)**2 ),dtype = float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ''' 这种写法jit不支持。jit支持的语法包括：https://numba.pydata.org/numba-doc/latest/reference/pysupported.html#pysupported
            aa = [[a[ii][jj] if ii >= 0 and ii < a.shape[0] and jj >= 0 and jj < a.shape[1] else np.nan \
                for jj in range(j-radius, j+radius+1)]
                    for ii in range(i-radius, i+radius+1)] #ii 表示行，在外，#+1是因为取不到
            b[i,j,:] = np.array(aa).reshape(-1)
            '''
            n = 0
            for ii in range(i-radius, i+radius+1):
                for jj in range(j-radius, j+radius+1):
                    if ii >= 0 and ii < a.shape[0] and jj >= 0 and jj < a.shape[1]:
                        b[i,j,n] = a[ii][jj]
                    else:
                        b[i,j,n] = np.nan
                    n = n + 1            
                
    return b
    

if __name__ == '__main__':
    gfs_data_dir: str = "/home/ubuntu/data/gfs/0p25"
    era5_data_dir: str = "/home/ubuntu/data/era5/data/"
    start_time_list: List[datetime] = [parse(a.split("0p")[-1][3:]) for a in glob.glob(os.path.join(gfs_data_dir, "20??", "??", "??", "??"))]
    csv_save_dir: str = "/home/ubuntu/work/MOS/train_csv/"
    os.makedirs(csv_save_dir, exist_ok=True)

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
        prepare_csv,
        gfs_varname_dict=var_short_dict_gfs,
        era5_varname_dict=var_short_dict_era5,
        gfs_data_dir=gfs_data_dir,
        era5_data_dir=era5_data_dir,
        save_dir=csv_save_dir) #partial:固定函数的一些参数

    job_list = []
    for dtime in start_time_list:
        for ifcst in range(3, 24+1, 3):
            job_list.append((dtime, ifcst))
            #[(datetime.datetime(2022, 9, 1, 6, 0), 3),  (datetime.datetime(2022, 9, 1, 6, 0), 6), ...]

    njobs: int = multiprocessing.cpu_count()

    with Pool(njobs) as p:
        p.starmap(save_func, job_list)

    #no jit 2m 32s
    #with jit 1m 30s

