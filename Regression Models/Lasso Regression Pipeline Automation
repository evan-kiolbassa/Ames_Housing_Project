import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from pandas.api.types import is_numeric_dtype
def pipeline_implementation():
    # Loading datasets and creating dummified variables for regression input
    ames_df = pd.read_csv(r'C:\Users\mmotd\OneDrive\Documents\Boot Camp Files\Machine Learning Project\Ames_Housing_Price_Data.csv')
    engineered_features = pd.read_csv(r'C:\Users\mmotd\OneDrive\Documents\Boot Camp Files\Machine Learning Project\Brian_Evan_Data.csv',
                                 usecols = ['airport_dist', 'downtown_dist', 'isu_dist'])
    # Dropping neighborhood features due to discrepancies in observation sizes leading to inaccurate coefficients
    ames_df = ames_df.drop('Neighborhood', axis = 1)
    columns = pd.Series(ames_df.columns)
    
    # Creating a list of rows where null values means there is simply no value
    regex_list = pd.Series(['(Alley)', '(Bsmt)', '(Fireplace)', '(Garage)', '(Pool)', '(Fence)', '(Misc)'])
    null_is_none = pd.Series(['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                    'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                    'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'])
    # Replacing NaN values with 'None' where the feature does not exist
    ames_df.loc[:,null_is_none] = ames_df.loc[:,null_is_none].apply(lambda x: x.replace(np.nan, 'None'))
    

    ordinals = ['LandSlope','ExterQual','ExterCond','HeatingQC','KitchenQual','FireplaceQu','GarageCond','PavedDrive',
                'LotShape', 'BsmtQual','BsmtCond','GarageQual','PoolQC','BsmtExposure','BsmtFinType1','BsmtFinType2',
                'CentralAir','GarageFinish','Functional','Street','Fence']
    
    ames_df = pd.concat([ames_df,engineered_features], axis=1)
    
    
    # Generating house and remodel age features and dropping columns that features are derived from
    ames_df['HouseAge'] = ames_df['YrSold'] - ames_df['YearBuilt']
    ames_df['RemodelAge'] = ames_df['YrSold'] - ames_df['YearRemodAdd']
    ames_df = ames_df.drop(['YrSold', 'YearBuilt', 'YearRemodAdd'], axis = 1)
    
    X = ames_df.drop(['SalePrice','PID','Unnamed: 0', 'OverallQual', 
                         'OverallCond'], axis = 1)
    categorical = [column for column in X.columns.tolist() if (list(X.loc[:,column].unique()) == [0,1])
                  | (list(X.loc[:,column].unique()) == [1])]
    
    
    X['HouseArea'] = X.loc[:, ['GrLivArea', 'MasVnrArea', 'TotalBsmtSF', 'GarageArea',
                               'WoodDeckSF', 'OpenPorchSF','PoolArea']].sum(axis=1)

    X = X.drop(['GrLivArea', 'MasVnrArea', 'TotalBsmtSF', 'GarageArea',
                               'WoodDeckSF', 'OpenPorchSF','PoolArea'], axis = 1)
    # Creating list of continuous numerical features to log transfmorm
    columns = X.columns
    continuous = [colname for colname in columns if is_numeric_dtype(X[colname]) == True]
    
    X.loc[:, continuous] = X.loc[:, continuous].apply(lambda x: np.log10(x + 2))
    enc = OrdinalEncoder()
    X.loc[:,ordinals] = enc.fit_transform(X.loc[:,ordinals])
    X = pd.get_dummies(X)
    y = np.log10(ames_df['SalePrice'])
    
    model = linear_model.Lasso()
    imputer = IterativeImputer(imputation_order = 'random',
                               initial_strategy = 'median',
                               max_iter = 10, sample_posterior = False)

    imputation_pipeline = Pipeline([
                      ('imputer', imputer),
                      ('model', model)])
    params = {}
    params['model__alpha'] = [4e-15, 3e-12, 5e-10, 7e-9, 6e-8, 5e-7, 8e-3, 6e-2, 5e-2, 0.1]
    params['model__selection'] = ['cyclic']
    params['model__max_iter'] = [15000]
    params['model__tol'] = [0.003, 0.001]
    params['model__normalize'] = ['False']

    grid_cv = GridSearchCV(imputation_pipeline, params, cv = 3,refit = 'neg_mean_squared_error',
                                   scoring=['neg_mean_squared_error', 
                                            'neg_mean_absolute_error', 
                                            'r2'], n_jobs = -1)
    optimized_lasso = Pipeline([
                      ('robust', RobustScaler()),
                      ('grid_cv', grid_cv)
     ])

    optimized_lasso.fit(X,y)

    # Sorting cross validation results
    grid_validation = pd.DataFrame(grid_cv.cv_results_).sort_values(ascending = True, by = 'rank_test_neg_mean_squared_error')
    string = 'Negative Root Mean Absolute Error: {accuracy:.2f}'
    print(string.format(accuracy = grid_cv.best_score_))


    # Conversion of lasso coefficients for interpretability
    lasso_coefs = pd.DataFrame(grid_cv.best_estimator_['model'].coef_, columns = ['Lasso Coefficients'],index = X.columns)
    optimized_lasso_coefs = lasso_coefs[abs(lasso_coefs['Lasso Coefficients']) > 0]
    exponent_transform = optimized_lasso_coefs.copy()
    house_area_interp = exponent_transform.loc['HouseArea',:] - 1
    columns = pd.Series(exponent_transform.index)
    categorical_index = columns[~columns.isin(continuous)].values
    categorical_interp = 10**exponent_transform.loc[continuous_index,:] - 1
    ordinal_index = columns[columns.isin(ordinals)].values
    ordinal_interp = exponent_transform.loc[ordinal_index,:] * 100
    continuous_index = columns[columns.isin(continuous)].values
    continuous_interp = exponent_transform.loc[continuous_index,:].drop('HouseArea') * 100
    # Returns the cross validation object, trained pipeline object, and converted betas for insertion into visualizations
    return optimized_lasso, grid_cv, ordinal_interp, continuous_interp, categorical_interp, house_area_interp
