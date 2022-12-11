from datetime import datetime,timedelta
import os
from multiprocessing import Pool

import argparse  
import os  
import time  
from datetime import datetime, timedelta  
from multiprocessing.dummy import Pool  

import cdsapi  
from loguru import logger 

resolution:int = 25 # resolution

global save_dir

def download_gfs(gfs_input):
    '''
    gfs_input: list 
    gfs_input[0] : datetime
    gfs_input[1] : the forecast hour
    '''
    init_time:datetime = gfs_input[0]
    ifore:int = gfs_input[1]
    print("init_time: ",init_time,"ifore: ",ifore)
    init_year = str(init_time.year).zfill(4)
    init_month = str(init_time.month).zfill(2)
    init_day = str(init_time.day).zfill(2)
    init_hour = str(init_time.hour).zfill(2)

    link = f"s3://noaa-gfs-bdp-pds/gfs.{init_year}{init_month}{init_day}/{init_hour}/atmos/gfs.t{init_hour}z.pgrb2.0p{str(resolution).zfill(2)}.f{str(ifore).zfill(3)}"
    output_dir = os.path.join(save_dir,f"0p{str(resolution).zfill(2)}",str(init_year),str(init_month),str(init_day),str(init_hour))
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir,f"gfs.t{init_hour}z.pgrb2.0p{str(resolution).zfill(2)}.f{str(ifore).zfill(3)}")
    if resolution == 50:
        filesize_limit = 140*1024*1024 
    else: # 25
        filesize_limit = 600*1024*1024 

    if (not os.path.exists(output_path)) or (os.path.getsize(output_path) < filesize_limit):
        print(f"aws s3 cp --no-sign-request {link} {output_dir}")
        os.system(f'aws s3 cp --no-sign-request "{link}" {output_dir}')


def download_gfs_batch(start_time:datetime,end_time:datetime, n_core:int = 4):
    ''' download the gfs data '''
    gfs_input = []
    while start_time <= end_time:
        for ifcst in range(3, 24+1, 3): # here we select the 3-24 h forecast
            print("will download:", start_time, ifcst)
            gfs_input.append([start_time,ifcst])
        start_time += timedelta(hours = 6) 

    start = datetime.now()
    with Pool(processes = n_core) as p: # the CPUs number
        p.map(download_gfs,gfs_input)
    print("downloading", 'by multi-processes', datetime.now() - start)

def download_era5(start: str, end: str):  

    '''
    download era5 data
    eg. download_era5(start="20220901", end="20220928")
    #cost ~ 3 min for one-month data with 4 CPUs with AWS p2 instance
    '''

    start_time: datetime = datetime.strptime(start, "%Y%m%d")  # 开始时间  
    end_time: datetime = datetime.strptime(end, "%Y%m%d")  # 结束时间  
    logger.info(f"start download: {start_time:%Y-%m-%d} to {end_time:%Y-%m-%d}")  

    joblist: List[Tuple[datetime, str]] = []  # allocate an empty list, it's made of tuples (datetime:datetime, str:str)
    while start_time <= end_time:  
        for i in ["sl"]:  # 这里可以修改，只下载sl还是也下载pl(pressure level) ,["sl", ""pl""]
            joblist.append((start_time, i))  
        start_time += timedelta(days=1)  

    print([x[0] for x in joblist]) #嵌套列表取值

    # Note，max request is 12. See https://cds.climate.copernicus.eu/vision  
    # create 4 threads  
    with Pool(processes = 4) as p: #depends on how many CPUs your computer has
        jobs = [x[0] for x in joblist]
        p.map(down_sl ,jobs)

def not_exist_or_small(filename: str) -> bool:  
    # 如果文件不存在或小于特定大小:则返回True  
    if not os.path.exists(filename):  
        return True  
    else:  
        filesize = 2770000000 if filename.endswith("pl.grib") else 12400000 #the size is 12459600 by one single-time test
        if os.path.getsize(filename) < filesize:  
            return True  
        else:  
            return False  
            
def down_sl(download_day: datetime):  
    '''download single-level data in ERA5'''
    logger.info(f"Downloading single levels for {download_day:%Y-%m-%d}")  
    year: str = download_day.strftime("%Y")  
    month: str = download_day.strftime("%m")  
    day: str = download_day.strftime("%d")  
    filename: str = os.path.join(save_dir,"era5", "data", year, month, f"{download_day:%Y%m%d}_sl.grib")  
    
    os.makedirs(os.path.join(save_dir,"era5","data", year, month), exist_ok=True)

    if not_exist_or_small(filename):  
        r = c.retrieve(  
            "reanalysis-era5-single-levels",  
            {  
                "product_type": "reanalysis",  
                "format": "grib",  
                'variable': [  
                    '10m_u_component_of_wind', '10m_v_component_of_wind', 
                    '2m_temperature', 'mean_sea_level_pressure', 
                    'surface_pressure', 'total_precipitation',  
                    '2m_dewpoint_temperature'
                ],  
                'time': [  
                    '00:00', '01:00', '02:00',  
                    '03:00', '04:00', '05:00',  
                    '06:00', '07:00', '08:00',  
                    '09:00', '10:00', '11:00',  
                    '12:00', '13:00', '14:00',  
                    '15:00', '16:00', '17:00',  
                    '18:00', '19:00', '20:00',  
                    '21:00', '22:00', '23:00',  
                ],  
                "year": year,  
                "month": month,  
                "day": day,  
                "area": [90, 0, 0, 180],  
            },  
            filename
        )  

        r.delete()  
        time.sleep(5)  
    else:  
        logger.warning(f"{filename} already exists")  
        logger.success(f"successfully download single levels for {download_day:%Y-%m-%d}")  
    return 0


if __name__ == '__main__':
    save_dir = '/d/work/data'
    download_gfs_batch(datetime(2022,10,1,6), datetime(2022,10,1,6))
    download_era5(start="20221001", end="20221003")