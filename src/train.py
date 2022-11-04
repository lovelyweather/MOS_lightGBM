# should "pip install dask distributed graphviz" for visualization
import dask.dataframe as dd
from dask.distributed import Client

import cfg
import os
import pickle
import lightgbm as lgb
from dask import delayed
from dask_ml.model_selection import RandomizedSearchCV, GridSearchCV # 还没有用到，后面调整参数试试
#dask支持并行模型训练和预测：dask-ml。
from loguru import logger

def process_X_y(var_MOS : str = 't2m'):

    os.makedirs(cfg.model_dir, exist_ok=True)
    client = Client(n_workers = os.cpu_count())

    ddf = dd.read_csv(os.path.join(cfg.csv_save_dir,"20??","??","*.csv"))
    ddf.astype({"ifore": "int16"})
    ddf = ddf.drop(columns=['Unnamed: 0'])
    X = ddf.drop(columns=['t2m', 'r2', 'd2m', 'u10', 'v10', 'msl'])
    # target variable for MOS
    
    y = ddf[[var_MOS]]
    return X, y

def gen_model(X, y, var_MOS = 't2m', search_cv = None, is_save_model = True):
    #https://ml.dask.org/modules/generated/dask_ml.model_selection.train_test_split.html    
     
    
    X_train, y_train = X, y
    #

    model = lgb.DaskLGBMRegressor(n_jobs=os.cpu_count(), num_leaves: int = 31)  

    # grid search  hyperparameter tuning
    param_grid = {
        'task' : ['predict'],
        'boosting': ['gbdt' ],
        'objective': ['root_mean_squared_error'],
        'num_iterations': [  1500, 2000,5000  ],
    #    'learning_rate':[  0.05, 0.005 ],
        'num_leaves':[ 7, 31],
        'max_depth' :[ 10, 25],
        'min_data_in_leaf':[20, 100],
        'feature_fraction': [ 0.6, 0.8,  0.9]     
    }

    if search_cv == "Random":

        RandomizedSearchCV = RandomizedSearchCV(model, param_grid = param_grid, cv=3, n_jobs=-1, verbose=10)
        RandomizedSearchCV.fit(X_train, y_train)

        print('best params')
        print(RandomizedSearchCV.best_params_)
        print(RandomizedSearchCV.best_score_)

        if is_save_model:
            with open(os.path.join(cfg.model_dir, f"{var_MOS}_RandomCV_lgb.pkl"), "wb") as f:
                pickle.dump(RandomizedSearchCV, f)

            logger.success(f"finish save {var_MOS}_RandomCV model")

    elif search_cv == "Grid":
        grid_search  = GridSearchCV(model, param_grid = param_grid, cv=3, n_jobs=-1, verbose=10)
        
        grid_search.fit(X_train, y_train)

        print('best params')
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        if is_save_model:
            with open(os.path.join(cfg.model_dir, f"{var_MOS}_GridCV_lgb.pkl"), "wb") as f:
                pickle.dump(grid_search, f)

            logger.success(f"finish save {var_MOS}_GridCV model")

    else:
        model.fit(X, y)
        if is_save_model:
            with open(os.path.join(cfg.model_dir, f"{var_MOS}_lgb.pkl"), "wb") as f:
                pickle.dump(model, f)

            logger.success(f"finish save {var_MOS} model")

       

    