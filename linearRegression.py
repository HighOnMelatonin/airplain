from typing import Optional, Any
import numpy as np
import pandas as pd
import cleanNether as cn

def getFeaturesTargets(df: pd.DataFrame, 
                         featureNames: list[str], 
                         targetNames: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Gets the features and targets arrays from a DataFrame.

    Args:
        df (pd.DataFrame): The dataframe to get the features and target from.
        featureNames (list[str]): The names of the columns to use as features.
        targetNames (list[str]): The names of the columns to use as targets.

    Functionality:
        - Locates the features and targets columns from df and returns them as dataframes.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The features dataframe and the targets dataframe.
    '''
    features: pd.DataFrame = df.copy().loc[:,featureNames]
    dfTarget: pd.DataFrame = df.copy().loc[:, targetNames]
    
    return features, dfTarget

def splitData(dfFeature: pd.DataFrame, dfTarget: pd.DataFrame, 
               randomState: Optional[int]=None, 
               testSize: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Splits the feature and target dataframes for training and testing.
    
    Args:
        dfFeature (pd.DataFrame): Dataframe of the features.
        dfTarget (pd.DataFrame): Dataframe of the targets.
        randomState = None (Optional[int]): The seed to use for the random splitting of the data.
        testSize = 0.5 (float): The percentage of the initial data to use as test data.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training features, test features, training targets, and test targets.

    Functionality:
        - Generates the indices of the data to be used for testing
        - Creates the test data sets by locating the indices from the data
        - Creates the training data sets by removing the indices from the data
    '''
    
    np.random.seed(randomState)
    size: int = dfFeature.shape[0]
    test_indices: np.ndarray = np.random.choice(size, int(size*testSize), False)
    
    dfFeatureTest: pd.DataFrame = dfFeature.loc[test_indices,:]
    dfTargetTest: pd.DataFrame = dfTarget.loc[test_indices,:]
    dfFeatureTrain: pd.DataFrame = dfFeature.drop(index=test_indices) 
    dfFeatureTest: pd.DataFrame = dfTarget.drop(index=test_indices) 

    return dfFeatureTrain, dfFeatureTest, dfFeatureTest, dfTargetTest

def prepareFeature(npFeature: np.ndarray) -> np.ndarray:
    '''
    Prepare an array for linear regression.
    
    Args:
        npFeature (np.ndarray): The array to prepare.

    Functionality:
        - Converts the array to a np.ndarray if it is a pd.DataFrame
        - Concatenate a column of 1s to the left of the array.
    
    Returns:
        np.ndarray: The prepared array.
    '''
    if isinstance(npFeature, pd.DataFrame):
        array: np.ndarray = array.to_numpy()
    npFeature: np.ndarray = npFeature
    oneArray: np.ndarray = np.ones((npFeature.shape[0],1))
    result: np.ndarray = np.concatenate((oneArray, npFeature), axis=1)
    return result

def normalizeZ(array: np.ndarray, columnsMeans: Optional[np.ndarray]=None,
                columnsStds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform z-normalization on an array.

    Args:
        array (np.ndarray): The array to normalize.
        columnsMeans = None(Optional[np.ndarray]): The means to use for normalization.
        columnsStds = None (Optional[np.ndarray]): The stds to use for normalization.
        
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The normalized array, the means used for normalizing, and the stds used for normalizing.

    Functionality:
        - Converts the array to a np.ndarray if it is a pd.DataFrame
        - Sets columnsMeans and columnsStds to those of the array's columns if either is not entered.
        - Normalizes a copy of the array with z-normlization

    Raises:
        AssertionError: When the size of the means or stds passed in do not match the size of the array.
    '''
    assert columnsMeans is None or columnsMeans.shape == (1, array.shape[1])
    assert columnsStds is None or columnsStds.shape == (1, array.shape[1])

    if isinstance(array, pd.DataFrame):
        array: np.ndarray = array.to_numpy()

    out: np.ndarray = array.copy()
    if columnsMeans is None:
        columnsMeans: np.ndarray = array.mean(axis=0).reshape(1, -1)
    if columnsStds is None:
        columnsStds: np.ndarray = array.std(axis=0).reshape(1, -1)

    if 0.0 in columnsStds:
        columnsStds: pd.DataFrame = pd.DataFrame(columnsStds)
        columnsStds = columnsStds.map(lambda x: 1 if x==0.0 else x)
        columnsStds = columnsStds.to_numpy().reshape((1, array.shape[1]))
        #columnsStds = np.array(list(map(lambda x: 1 if np.all(x==0.0) else x,columnsStds)))
    

    try:
        out: np.ndarray = (out - columnsMeans) / columnsStds
    except:
        out: np.ndarray = (out - columnsMeans) / 1
    
    assert out.shape == array.shape
    assert columnsMeans.shape == (1, array.shape[1])
    assert columnsStds.shape == (1, array.shape[1])

    return out, columnsMeans, columnsStds

def calcLinreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
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

def computeCostLinreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
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
    yHat: np.ndarray = calcLinreg(X, beta)
    yDiff: np.ndarray = (yHat - y)
    diffSums: np.ndarray = np.matmul( yDiff.T, yDiff )
    J: np.ndarray = diffSums / ( 2 * X.shape[0] )
    assert J.shape == (1, 1)
    return np.squeeze(J)

def gradientDescentLinreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray,   
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
        yHat: np.ndarray = calcLinreg(X, beta)
        yDiff: np.ndarray = yHat - y
        diffSum: np.ndarray = np.matmul( X.T, yDiff )
        beta: np.ndarray = beta - (alpha / m * diffSum)
        J: np.ndarray = computeCostLinreg( X, y, beta )
        J_storage[i] = J
        
    assert beta.shape == (X.shape[1], 1)
    assert J_storage.shape == (num_iters, 1)
    return beta, J_storage

def buildModelLinreg(dfFeatureTrain: pd.DataFrame,
                       dfFeatureTest: pd.DataFrame,
                       beta: Optional[np.ndarray] = None,
                       alpha: float = 0.01,
                       iterations: int = 1500) -> tuple[dict[str, Any], np.ndarray]:
    '''
    Builds a linear regression model for the given features and targets.
    
    Args:
        dfFeatureTrain (pd.DataFrame): Dataframe of the training features.
        dfFeatureTest (pd.DataFrame): Dataframe of the training targets.
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
        beta = np.zeros((dfFeatureTrain.shape[1] + 1, 1)) 
    assert beta.shape == (dfFeatureTrain.shape[1] + 1, 1)

    model: dict[str, Any] = {}
    
    
    arrayFeatureTrain: np.ndarray = dfFeatureTrain.to_numpy()
    arrayFeatureTrainZ: np.ndarray; means: np.ndarray; stds: np.ndarray
    arrayFeatureTrainZ, means, stds = normalizeZ(arrayFeatureTrain)

    X = prepareFeature(arrayFeatureTrainZ)
    y: np.ndarray = dfFeatureTest.to_numpy()

    beta, J_storage = gradientDescentLinreg(X, y, beta, alpha, iterations)

    model['beta'], model['means'], model['stds'] = beta, means, stds
    
    assert model["beta"].shape == (dfFeatureTrain.shape[1] + 1, 1)
    assert model["means"].shape == (1, dfFeatureTrain.shape[1])
    assert model["stds"].shape == (1, dfFeatureTrain.shape[1])
    assert J_storage.shape == (iterations, 1)
    return model, J_storage

def predictLinreg(arrayFeature: np.ndarray, beta: np.ndarray, 
                   means: Optional[np.ndarray]=None, 
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    '''
    Gets the predicted straight line equation from the given features and coefficients.
    
    Args:
        arrayFeature (np.ndarray): The array of features.
        beta (np.ndarray): The coefficients found from linear regression. 
        means = None (Optional[np.ndarray]): The means to use for normalization.
        stds = None (Optional[np.ndarray]): The stds to use for normalization.

    Functionality:
        - Converts the array to a np.ndarray if it is a pd.DataFrame.
        - Sets columnsMeans and columnsStds to those of the array's columns if either is not entered.
        - Normalizes the array with z-normalization then prepares it for computation.
        - Computes the predicted target values.
    
    Returns:
        np.ndarray: The predicted values that form the straight line equation.
    
    Raises:
        AssertionError: When the size of the means or stds passed in do not match the size of the array.
    '''
    assert means is None or means.shape == (1, arrayFeature.shape[1])
    assert stds is None or stds.shape == (1, arrayFeature.shape[1])
    if isinstance(arrayFeature, pd.DataFrame):
        arrayFeature: np.ndarray = arrayFeature.to_numpy()
    X, _, _ = normalizeZ(arrayFeature, means, stds)
    X = prepareFeature(X)
    result: np.ndarray = calcLinreg(X, beta)
    
    assert result.shape == (arrayFeature.shape[0], 1)
    return result

def r2Score(y: np.ndarray, ypred: np.ndarray) -> float:
    '''
    Calculates the r2 score of the linear regression.
    
    Args:
        y (np.ndarray): The true target values.
        ypred (np.ndarray): The predicted target values.
    
    Returns:
        float: The calculated r2 score.

    Functionality:
        - Computes ( 1 - (ssRes / ss_tot) ). 
    '''
    yDiff: np.ndarray = y - ypred
    ssRes: np.ndarray = np.sum( yDiff.T @ yDiff )
    yMeanDiff: np.ndarray = y - y.mean()
    ss_tot: np.ndarray = np.sum( yMeanDiff.T @ yMeanDiff )
    return float(1 - (ssRes / ss_tot))

def meanSquaredError(target: np.ndarray, pred: np.ndarray) -> float:
    '''
    Calculates the mean squared error of the linear regression.
    
    Args:
        target (np.ndarray): The true target values.
        pred (np.ndarray): The predicted target values.
    
    Returns:
        float: The calculated mean squared error.

    Functionality:
        - Computes 1 / n * ( y_i - yHat_i) ** 2
    '''

    yDiff: np.ndarray = target - pred
    total: np.ndarray = np.sum(yDiff.T @ yDiff)
    n: int = target.shape[0]
    return float(total / n)




if __name__ == '__main__':

    #Grab the cleaned data
    print('-'*150)
    popDensityDF: pd.DataFrame = pd.read_csv('datafiles/processedPopDensity.csv',index_col=0).fillna(-1)
    # pmDF: pd.DataFrame = pd.read_csv('datafiles/processedPm2.5.csv',index_col=0).fillna(-1)
    proximityDF: pd.DataFrame = pd.read_csv('datafiles/processedProximity.csv')
    landUseDF: pd.DataFrame = pd.read_csv('datafiles/processedLandUse.csv',sep=";",index_col=0).fillna(-1) 
    transportPublicDF: pd.DataFrame = pd.read_csv('datafiles/processedCarTravelPublic.csv')
    transportPrivateDF: pd.DataFrame = pd.read_csv('datafiles/processedCarTravelPrivate.csv')
    print('-'*150)
    print(f'{popDensityDF=}')
    # print(f'{pmDF=}')
    print(f'{proximityDF=}')
    print(f'{landUseDF=}')
    print(f'{transportPrivateDF=}')
    print(f'{transportPublicDF=}')
    print('-'*150)

    #Find the row that has the specified region and year, then return the target column in that row
    def getDataFromDF(df: pd.DataFrame, region: str, year: str, targetCol: str) -> str:
        '''
        Descripton.
        
        Args:
            some_arg (Any):
        
        Returns:
            Any:
        
        Functionality
            - does a thing
        
        Raises:
            AssertionError:
        '''
        #print(f'{region=}, {type(region)=}, {year=}, {type(year)=}')
        #print(year, type(year))
        year = int(year)
        regionMatches: pd.Series = df["Region"] == region
        yearMatches: pd.Series = df["Year"] == year
        regionMatches: set = set(df.index[regionMatches].tolist())
        yearMatches: set = set(df.index[yearMatches].tolist())
        desiredIndices: set = regionMatches & yearMatches
        #print(f'{len(regionMatches)=}, {len(yearMatches)=}')
        #print(f'{len(desiredIndices)=}')
        assert len(desiredIndices) == 1
        desiredIndex: int = list(desiredIndices)[0]
        targetIndex: int = df.columns.get_loc(targetCol)

        output = df.iloc[desiredIndex,targetIndex]
        return output



        

    #Merge the cleaned data into 1 dataframe
    newColumns: pd.DataFrame = pd.DataFrame(columns=['Population Density', 'Total Road Area', 'Total Greenery Area','Public Transport Travel','Private Transport Travel'])
    for index, row in proximityDF.iterrows():
        region: str = row['Region']
        year: str = str(row['Year'])

        #Process population density
        try:
            popDensity: float = popDensityDF.loc[region,year]
        except:
            popDensity: float = -1.0

        # try:
        #     pm25: float = pmDF.loc[region,year]
        #     if "-" in pm25:
        #         pm25: float = -1.0
        #     if "*" in pm25:
        #         pm25: float = pm25.strip().replace(" *","")
        # except:
        #     pm25: float = -1.0
        # if isinstance(pm25, pd.Series):
        #     finalValue: float = 0
        #     isDash: bool = False
        #     for val in pm25:
        #         val = val.strip()
        #         if val == "-":
        #             isDash: bool = True
        #             break
        #         val = val.replace(" *","")
        #         finalValue += float(val)
        #     pm25: float = finalValue / len(pm25)
        #     if isDash:
        #         pm25: float = -1.0

        #Process land use
        try:
            roadArea: float | str; greenArea: float
            value = landUseDF.loc[region, year]
            assert value != -1.0
            value = value.replace("(","").replace(")","").replace('\'','')
            roadArea, greenArea = value.split(",")
            roadArea = float(roadArea)
            greenArea = float(greenArea)
            print(roadArea, greenArea)

        except:
            roadArea: float = -1.0
            greenArea: float = -1.0

        #Process travel data
        try:
            publicTransport: float = float(getDataFromDF(transportPublicDF,region,year, 'Public Transport in km'))
            #print('found')
        except:
            publicTransport: float = -1.0
        try:
            privateTransport: float = float(getDataFromDF(transportPrivateDF,region,year, 'Private Transport in km'))
            #print('found')
        except:
            privateTransport: float = -1.0
        

        # print(f'{popDensity=}, {pm25=}, {roadArea=}, {greenArea=}')
        newColumns.loc[index] = [popDensity, roadArea, greenArea, publicTransport, privateTransport]
    mergedDF: pd.DataFrame = pd.concat([proximityDF,newColumns],axis=1)
    print(mergedDF)
    mergedDF.to_csv('compiledData.csv',index=False)

    print('-'*150)
    #Remove incomplete rows from the data to get the final compiled dataset
    incompleteRows: list = []
    for index, row in mergedDF.iterrows():
        complete: bool = True
        # '0 to 10 km','>10 to 20 km','>20 to 50km', 'Population Density','Total Road Area','Total Greenery Area', 'Public Transport Travel', 'Private Transport Travel' For testing
        whiteList : list[str] = ['Population Density','Public Transport Travel','Private Transport Travel'] #For testing
        for key, value in row.items():
            if not key in whiteList: #For testing
                continue #For testing
            if value == -1.0:
                complete: bool = False
                break
        if not complete:
            incompleteRows.append(index)
    data: pd.DataFrame = mergedDF.drop(incompleteRows, axis=0).reset_index()
    print('-'*150)
    print(data)

    def convertStrToFloat(value: str) -> str | float:
        try:
            value: float = float(value)
        except:
            pass
        return value
    data = data.map(convertStrToFloat)
    data = data.map(lambda x: x if x != -1.0 else 0.0)


    ### Linear regression, don't run until dataset is fixed
    # features = ['0 to 10 km', '>10 to 20 km', '>20 to 50km', 'Population Density']
    # target = ['PM2.5']
    # dataFeatures, dataTarget = getFeaturesTargets(data, features, target) 
    # dataFeaturesTrain, dataFeaturesTest, dataTarget_train, dataTarget_test = splitData(dataFeatures, dataTarget, randomState=100, testSize=0.3)
    # model, J_storage = buildModelLinreg(dataFeaturesTrain, dataTarget_train)
    # pred: np.ndarray = predictLinreg(dataFeaturesTest.to_numpy(), model['beta'], model['means'], model['stds'])

    # import matplotlib.pyplot as plt
    # import matplotlib.axes as axes

    # print('-'*150)
    # print(f'{model["beta"]=}')
    # print(f'{model["means"]=}')
    # print(f'{model["stds"]=}')


    # for feature in features:
    #     plt.scatter(dataFeaturesTest[feature], dataTarget_test)
    #     plt.scatter(dataFeaturesTest[feature], pred)


