# should "pip install dask distributed graphviz" for visualization
import dask.dataframe as dd
from dask.distributed import Client

import os
import pickle
import lightgbm as lgb
from dask import delayed
from dask_ml.model_selection import RandomizedSearchCV # 还没有用到，后面调整参数试试
#dask支持并行模型训练和预测：dask-ml。
from loguru import logger

# dask_ml.model_selection.train_test_split 分测试集这些也还没用到


import os


os.makedirs(model_dir, exist_ok=True)

def process_X_y(var_MOS : str = 't2m'):

    client = Client(n_workers = os.cpu_count())

    ddf = dd.read_csv(os.path.join(csv_dir ,"20??","??","*.csv"))
    ddf.astype({"ifore": "int16"})
    ddf = ddf.drop(columns=['Unnamed: 0'])
    X = ddf.drop(columns=['t2m', 'r2', 'd2m', 'u10', 'v10', 'msl'])
    # target variable for MOS
    
    y = ddf[[var_MOS]]
    return X, y

#https://ml.dask.org/modules/generated/dask_ml.model_selection.train_test_split.html
X, y = process_X_y()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0, test_size = 0.2)

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

RandomizedSearchCV = GridSearchCV(model, param_grid = param_grid, cv=3, n_jobs=-1, verbose=10)
RandomizedSearchCV.fit(X_train, y_train)
print('best params')
print (RandomizedSearchCV.best_params_)
print(RandomizedSearchCV.best_score_)

with open(os.path.join(model_dir, f"{var_MOS}_lgb.pkl"), "wb") as f:
        pickle.dump(model, f)
    # returning the trained model
    logger.success(f"finish save {era_var} model")


y_pred = RandomizedSearchCV.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred))
print(" RMSE: %f" % (rmse_lgb ))