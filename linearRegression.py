from typing import Optional, Any
import numpy as np
import pandas as pd
import cleanNether as cn

def get_features_targets(df: pd.DataFrame, 
                         feature_names: list[str], 
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Gets the features and targets arrays from a DataFrame.

    Args:
        df (pd.DataFrame): The dataframe to get the features and target from.
        feature_names (list[str]): The names of the columns to use as features.
        target_names (list[str]): The names of the columns to use as targets.

    Functionality:
        - Locates the features and targets columns from df and returns them as dataframes.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The features dataframe and the targets dataframe.
    '''
    features: pd.DataFrame = df.copy().loc[:,feature_names]
    df_target: pd.DataFrame = df.copy().loc[:, target_names]
    
    return features, df_target

def split_data(df_feature: pd.DataFrame, df_target: pd.DataFrame, 
               random_state: Optional[int]=None, 
               test_size: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Splits the feature and target dataframes for training and testing.
    
    Args:
        df_feature (pd.DataFrame): Dataframe of the features.
        df_target (pd.DataFrame): Dataframe of the targets.
        random_state = None (Optional[int]): The seed to use for the random splitting of the data.
        test_size = 0.5 (float): The percentage of the initial data to use as test data.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training features, test features, training targets, and test targets.

    Functionality:
        - Generates the indices of the data to be used for testing
        - Creates the test data sets by locating the indices from the data
        - Creates the training data sets by removing the indices from the data
    '''
    
    np.random.seed(random_state)
    size: int = df_feature.shape[0]
    test_indices: np.ndarray = np.random.choice(size, int(size*test_size), False)
    
    df_feature_test: pd.DataFrame = df_feature.loc[test_indices,:]
    df_target_test: pd.DataFrame = df_target.loc[test_indices,:]
    df_feature_train: pd.DataFrame = df_feature.drop(index=test_indices) 
    df_target_train: pd.DataFrame = df_target.drop(index=test_indices) 

    return df_feature_train, df_feature_test, df_target_train, df_target_test

def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    '''
    Prepare an array for linear regression.
    
    Args:
        np_feature (np.ndarray): The array to prepare.

    Functionality:
        - Converts the array to a np.ndarray if it is a pd.DataFrame
        - Concatenate a column of 1s to the left of the array.
    
    Returns:
        np.ndarray: The prepared array.
    '''
    if isinstance(np_feature, pd.DataFrame):
        array: np.ndarray = array.to_numpy()
    np_feature: np.ndarray = np_feature
    one_array: np.ndarray = np.ones((np_feature.shape[0],1))
    result: np.ndarray = np.concatenate((one_array, np_feature), axis=1)
    return result

def normalize_z(array: np.ndarray, columns_means: Optional[np.ndarray]=None,
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform z-normalization on an array.

    Args:
        array (np.ndarray): The array to normalize.
        columns_means = None(Optional[np.ndarray]): The means to use for normalization.
        columns_stds = None (Optional[np.ndarray]): The stds to use for normalization.
        
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The normalized array, the means used for normalizing, and the stds used for normalizing.

    Functionality:
        - Converts the array to a np.ndarray if it is a pd.DataFrame
        - Sets columns_means and columns_stds to those of the array's columns if either is not entered.
        - Normalizes a copy of the array with z-normlization

    Raises:
        AssertionError: When the size of the means or stds passed in do not match the size of the array.
    '''
    assert columns_means is None or columns_means.shape == (1, array.shape[1])
    assert columns_stds is None or columns_stds.shape == (1, array.shape[1])

    if isinstance(array, pd.DataFrame):
        array: np.ndarray = array.to_numpy()

    out: np.ndarray = array.copy()
    if columns_means is None:
        columns_means: np.ndarray = array.mean(axis=0).reshape(1, -1)
    if columns_stds is None:
        columns_stds: np.ndarray = array.std(axis=0).reshape(1, -1)

    out: np.ndarray = (out - columns_means) / columns_stds
    
    assert out.shape == array.shape
    assert columns_means.shape == (1, array.shape[1])
    assert columns_stds.shape == (1, array.shape[1])

    return out, columns_means, columns_stds

def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    '''
    Calculates the resulting values from the linear regression.
    
    Args:
        X (np.ndarray): The array of feature values.
        beta (np.ndarray): The coefficients found from linear regression.

    Returns:
        np.ndarray: The resulting values from the features and betas.

    Functionality:
        - Matrix multiplies X and beta.
    
    Raises:
        AssertionError: When the shape of beta does not correspond to the shape of X.
    '''
    assert beta.shape[1] == 1
    result: np.ndarray = np.matmul(X, beta)
    assert result.shape == (X.shape[0], 1)
    return result

def compute_cost_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    '''
    Computes the cost function for linear regression.
    
    Args:
        X (np.ndarray): The features values.
        y (np.ndarray): The target values.
        beta (np.ndarray): The current predicted coefficients.
    
    Returns:
        np.ndarray: The resulting cost.

    Functionality:
        - Compute ( 1 / ( 2 * m ) * ( y_x_i - y_i ) ** 2 )
    
    Raises:
        AssertionError: When the resulting cost is off the wrong shape.
    '''
    y_hat: np.ndarray = calc_linreg(X, beta)
    y_diff: np.ndarray = (y_hat - y)
    diff_sums: np.ndarray = np.matmul( y_diff.T, y_diff )
    J: np.ndarray = diff_sums / ( 2 * X.shape[0] )
    assert J.shape == (1, 1)
    return np.squeeze(J)

def gradient_descent_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray,   
                            alpha: float, num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute the coefficients for linear regression via the gradient descent algorithim.
    
    Args:
        X (np.ndarray): The features values.
        y (np.ndarray): The target values.
        beta (np.ndarray): The initial coefficients to use.
        alpha (float): The learning rate for the algorithim.
        num_iters (int): The number of iterations to run the algorithim.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: An array of final predicted coefficients, and an array of the costs from each iteration of the algorithim.

    Functionality:
        - Computes the new beta values and stores the cost for the given number of iterations.
    
    Raises:
        AssertionError: When the shapes of X, y, or beta do not align with each other.
    '''
    
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
    '''
    Builds a linear regression model for the given features and targets.
    
    Args:
        df_feature_train (pd.DataFrame): Dataframe of the training features.
        df_target_train (pd.DataFrame): Dataframe of the training targets.
        beta = None (Optional[np.ndarray]): The initial predicted coefficients for the linear regression
        alpha = 0.01 (float): The learning rate for the gradient descent algorithim.
        iterations = 1500 (int): The number of iterations to run the gradient descent algorithim.
    
    Returns:
        tuple[dict[str, Any], np.ndarray]: The linear regression model, and an array of the costs from each iteration of the algorithim. The model contains 'beta': the final predicted coefficients, 'means': the means used to normalize the feature data, and 'stds': the stds used to normalize the feature data.

    Functionality:
        - If no beta is entered, sets it to an array of 0s of the appropiate shape.
        - Normalizes and prepares the feature data, prepares the target data.
        - Performs linear regression with the gradient descent algorithim, and stores the resulting data into a dictionary.
    
    Raises:
        AssertionError: When the shape of beta does not align with the shape of the feature and target array shapes.
    '''

    if beta is None:
        beta = np.zeros((df_feature_train.shape[1] + 1, 1)) 
    assert beta.shape == (df_feature_train.shape[1] + 1, 1)

    model: dict[str, Any] = {}
    
    
    array_feature_train: np.ndarray = df_feature_train.to_numpy()
    array_feature_train_z: np.ndarray; means: np.ndarray; stds: np.ndarray
    array_feature_train_z, means, stds = normalize_z(array_feature_train)

    X = prepare_feature(array_feature_train_z)
    y: np.ndarray = df_target_train.to_numpy()

    beta, J_storage = gradient_descent_linreg(X, y, beta, alpha, iterations)

    model['beta'], model['means'], model['stds'] = beta, means, stds
    
    assert model["beta"].shape == (df_feature_train.shape[1] + 1, 1)
    assert model["means"].shape == (1, df_feature_train.shape[1])
    assert model["stds"].shape == (1, df_feature_train.shape[1])
    assert J_storage.shape == (iterations, 1)
    return model, J_storage

def predict_linreg(array_feature: np.ndarray, beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    '''
    Gets the predicted straight line equation from the given features and coefficients.
    
    Args:
        array_feature (np.ndarray): The array of features.
        beta (np.ndarray): The coefficients found from linear regression. 
        means = None (Optional[np.ndarray]): The means to use for normalization.
        stds = None (Optional[np.ndarray]): The stds to use for normalization.

    Functionality:
        - Converts the array to a np.ndarray if it is a pd.DataFrame.
        - Sets columns_means and columns_stds to those of the array's columns if either is not entered.
        - Normalizes the array with z-normalization then prepares it for computation.
        - Computes the predicted target values.
    
    Returns:
        np.ndarray: The predicted values that form the straight line equation.
    
    Raises:
        AssertionError: When the size of the means or stds passed in do not match the size of the array.
    '''
    assert means is None or means.shape == (1, array_feature.shape[1])
    assert stds is None or stds.shape == (1, array_feature.shape[1])
    if isinstance(array_feature, pd.DataFrame):
        array_feature: np.ndarray = array_feature.to_numpy()
    X, _, _ = normalize_z(array_feature, means, stds)
    X = prepare_feature(X)
    result: np.ndarray = calc_linreg(X, beta)
    
    assert result.shape == (array_feature.shape[0], 1)
    return result

def r2_score(y: np.ndarray, ypred: np.ndarray) -> float:
    '''
    Calculates the r2 score of the linear regression.
    
    Args:
        y (np.ndarray): The true target values.
        ypred (np.ndarray): The predicted target values.
    
    Returns:
        float: The calculated r2 score.

    Functionality:
        - Computes ( 1 - (ss_res / ss_tot) ). 
    '''
    y_diff: np.ndarray = y - ypred
    ss_res: np.ndarray = np.sum( y_diff.T @ y_diff )
    y_mean_diff: np.ndarray = y - y.mean()
    ss_tot: np.ndarray = np.sum( y_mean_diff.T @ y_mean_diff )
    return float(1 - (ss_res / ss_tot))

def mean_squared_error(target: np.ndarray, pred: np.ndarray) -> float:
    '''
    Calculates the mean squared error of the linear regression.
    
    Args:
        target (np.ndarray): The true target values.
        pred (np.ndarray): The predicted target values.
    
    Returns:
        float: The calculated mean squared error.

    Functionality:
        - Computes 1 / n * ( y_i - y_hat_i) ** 2
    '''

    y_diff: np.ndarray = target - pred
    total: np.ndarray = np.sum(y_diff.T @ y_diff)
    n: int = target.shape[0]
    return float(total / n)




if __name__ == '__main__':
    print(cn.translateRegionCode())
    cn.trimRegionCode("reregionCode")
    print(cn.popDensity())
    print(cn.trimPM())
    print(cn.trimProximity())
    #Grab the cleaned data
    popDensityDF: pd.DataFrame = pd.read_csv('datafiles\processed_pop_density.csv',index_col=0).fillna(-1)
    pmDF: pd.DataFrame = pd.read_csv('datafiles\processed_pm2.5.csv',index_col=0).fillna(-1)
    proximityDF: pd.DataFrame = pd.read_csv('datafiles\processed_proximity.csv')
    print('-'*150)
    print(popDensityDF)
    print(pmDF)
    print(proximityDF)

    print('-'*150)
    #Merge the cleaned data into 1 dataframe
    newColumns: pd.DataFrame = pd.DataFrame(columns=['Population Density','PM2.5'])
    for index, row in proximityDF.iterrows():
        region: str = row['Region']
        year: str = str(row['Year'])
        try:
            popDensity: float = popDensityDF.loc[region,year]
        except:
            popDensity: float = -1.0
        try:
            pm25: float = pmDF.loc[region,year]
        except:
            pm25: float = -1.0
        newColumns.loc[index] = [popDensity, pm25]
    mergedDF: pd.DataFrame = pd.concat([proximityDF,newColumns],axis=1)
    print(mergedDF)

    print('-'*150)
    #Remove incomplete rows from the data to get the final compiled dataset
    incompleteRows: list = []
    for index, row in mergedDF.iterrows():
        complete: bool = True
        for key, value in row.items():
            ### Exclude 'PM2.5' from the incomplete check, for testing purposes. Remove for submission.
            if key == 'PM2.5':
                continue
            if value == -1.0:
                complete: bool = False
                break
        if not complete:
            incompleteRows.append(index)
    data: pd.DataFrame = mergedDF.drop(incompleteRows, axis=0).reset_index()
    print('-'*150)
    print(data)
    ### Linear regression, don't run until dataset is fixed
    # features = ['0 to 10 km', '>10 to 20 km', '>20 to 50km', 'Population Density']
    # target = ['PM2.5']
    # data_features, data_target = get_features_targets(data, features, target) 
    # data_features_train, data_features_test, data_target_train, data_target_test = split_data(data_features, data_target, random_state=100, test_size=0.3)
    # model, J_storage = build_model_linreg(data_features_train, data_target_train)
    # pred: np.ndarray = predict_linreg(data_features_test.to_numpy(), model['beta'], model['means'], model['stds'])

    # import matplotlib.pyplot as plt
    # import matplotlib.axes as axes

    # print('-'*150)
    # print(f'{model["beta"]=}')
    # print(f'{model["means"]=}')
    # print(f'{model["stds"]=}')


    # for feature in features:
    #     plt.scatter(data_features_test[feature], data_target_test)
    #     plt.scatter(data_features_test[feature], pred)


