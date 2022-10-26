# download data
save_dir = '/d/work/data'


# prepare csv
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

# split dataset, train and tuning


# predict!





