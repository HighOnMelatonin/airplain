from typing import Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns

def convert_df_to_np(array: np.ndarray | pd.DataFrame) -> np.ndarray: # type: ignore
    if isinstance(array, pd.DataFrame):
        array: np.ndarray = array.to_numpy()
    return array

def normalize_z(array: np.ndarray, columns_means: Optional[np.ndarray]=None,  # type: ignore
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # type: ignore
    assert columns_means is None or columns_means.shape == (1, array.shape[1])
    assert columns_stds is None or columns_stds.shape == (1, array.shape[1])

    out: np.ndarray = convert_df_to_np(array).copy()
    if columns_means is None:
        columns_means: np.ndarray = array.mean(axis=0).reshape(1, -1)
    if columns_stds is None:
        columns_stds: np.ndarray = array.std(axis=0).reshape(1, -1)

    out: np.ndarray = (out - columns_means) / columns_stds
    
    assert out.shape == array.shape
    assert columns_means.shape == (1, array.shape[1])
    assert columns_stds.shape == (1, array.shape[1])

    return out, columns_means, columns_stds

def get_features_targets(df: pd.DataFrame, 
                         feature_names: list[str], 
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    features: pd.DataFrame = df.copy().loc[:,feature_names]
    df_target: pd.DataFrame = df.copy().loc[:, target_names]
    
    return features, df_target

def prepare_feature(np_feature: np.ndarray) -> np.ndarray: # type: ignore
    np_feature: np.ndarray = convert_df_to_np(np_feature)
    one_array: np.ndarray = np.ones((np_feature.shape[0],1))
    result: np.ndarray = np.concatenate((one_array, np_feature), axis=1)
    return result

def predict_linreg(array_feature: np.ndarray, beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    assert means is None or means.shape == (1, array_feature.shape[1])
    assert stds is None or stds.shape == (1, array_feature.shape[1])
    X: np.ndarray = convert_df_to_np(array_feature)
    X, _, _ = normalize_z(X, means, stds)
    X = prepare_feature(X)
    result: np.ndarray = calc_linreg(X, beta)
    
    assert result.shape == (array_feature.shape[0], 1)
    return result

def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    assert beta.shape[1] == 1
    result: np.ndarray = np.matmul(X, beta)
    assert result.shape == (X.shape[0], 1)
    return result

def split_data(df_feature: pd.DataFrame, df_target: pd.DataFrame, 
               random_state: Optional[int]=None, 
               test_size: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    np.random.seed(random_state)
    size: int = df_feature.shape[0]
    test_indices: np.ndarray = np.random.choice(size, int(size*test_size), False)
    
    df_feature_test: pd.DataFrame = df_feature.loc[test_indices,:]
    df_target_test: pd.DataFrame = df_target.loc[test_indices,:]
    df_feature_train: pd.DataFrame = df_feature.drop(index=test_indices) # type: ignore
    df_target_train: pd.DataFrame = df_target.drop(index=test_indices) # type: ignore

    return df_feature_train, df_feature_test, df_target_train, df_target_test

def r2_score(y: np.ndarray, ypred: np.ndarray) -> float:
    y_diff: np.ndarray = y - ypred
    ss_res: np.ndarray = np.sum( y_diff.T @ y_diff )
    y_mean_diff: np.ndarray = y - y.mean()
    ss_tot: np.ndarray = np.sum( y_mean_diff.T @ y_mean_diff )
    return float(1 - (ss_res / ss_tot))

def mean_squared_error(target: np.ndarray, pred: np.ndarray) -> float:

    y_diff: np.ndarray = target - pred
    total: np.ndarray = np.sum(y_diff.T @ y_diff)
    n: int = target.shape[0]
    return float(total / n)

def compute_cost_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    y_hat: np.ndarray = calc_linreg(X, beta)
    y_diff: np.ndarray = (y_hat - y)
    diff_sums: np.ndarray = np.matmul( y_diff.T, y_diff )
    J: np.ndarray = diff_sums / ( 2 * X.shape[0] )
    assert J.shape == (1, 1)
    return np.squeeze(J)

def gradient_descent_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray,   # type: ignore
                            alpha: float, num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    
    m: int = X.shape[0]
    J_storage: np.ndarray = np.zeros( (num_iters, 1) )
        
    for i in range(num_iters):
        y_hat: np.ndarray = calc_linreg(X, beta)
        y_diff: np.ndarray = y_hat - y
        diff_sum: np.ndarray = np.matmul( X.T, y_diff )
        beta: np.ndarray = beta - (alpha / m * diff_sum)
        J: np.ndarray = compute_cost_linreg( X, y, beta )
        J_storage[i] = J
        
    assert beta.shape == (X.shape[1], 1)
    assert J_storage.shape == (num_iters, 1)
    return beta, J_storage

def build_model_linreg(df_feature_train: pd.DataFrame,
                       df_target_train: pd.DataFrame,
                       beta: Optional[np.ndarray] = None,
                       alpha: float = 0.01,
                       iterations: int = 1500) -> tuple[dict[str, Any], np.ndarray]:
    if beta is None:
        beta = np.zeros((df_feature_train.shape[1] + 1, 1)) 
    assert beta.shape == (df_feature_train.shape[1] + 1, 1)

    model: dict[str, Any] = {}
    
    array_feature_train: np.ndarray = convert_df_to_np(df_feature_train)

    array_feature_train_z: np.ndarray; means: np.ndarray; stds: np.ndarray
    array_feature_train_z, means, stds = normalize_z(array_feature_train)

    X = prepare_feature(array_feature_train_z)
    y: np.ndarray = convert_df_to_np(df_target_train)

    beta, J_storage = gradient_descent_linreg(X, y, beta, alpha, iterations)

    model['beta'], model['means'], model['stds'] = beta, means, stds
    
    assert model["beta"].shape == (df_feature_train.shape[1] + 1, 1)
    assert model["means"].shape == (1, df_feature_train.shape[1])
    assert model["stds"].shape == (1, df_feature_train.shape[1])
    assert J_storage.shape == (iterations, 1)
    return model, J_storage

