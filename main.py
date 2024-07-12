# base
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Важная настройка для корректной настройки pipeline!
import sklearn
sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import GridSearchCV, KFold

from category_encoders import TargetEncoder

# for model learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

#models
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_predict

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# tunning hyperparamters model
import optuna

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Название
st.title('Определение цен на недвижимость')
# Описание
st.write('< < < Загрузите датафреймы train.csv и test.csv для определения цены')
# Шаг 1. Загрузка CSV файла
uploaded_train = st.sidebar.file_uploader('Загрузи CSV файл train.csv', type='csv')
uploaded_test = st.sidebar.file_uploader('Загрузи CSV файл test.csv', type='csv')


if uploaded_train and uploaded_test:
    # Считывание данных из файлов
    X_train = pd.read_csv(uploaded_train)
    X_test = pd.read_csv(uploaded_test)
    # Вывод тренирировочного датасета
    st.write(X_train.head(5))

    # Выделение таргета в отдельную переменную
    X_train, y = X_train.drop('SalePrice', axis=1), X_train['SalePrice']

    # Объединение train и test в один фрейм для заполнения пропусков и обработки данных
    X = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Группировка фич на фичи на удаление, на числовые, на категориальные и на фичи для ковеерной замены пустых значений
    drop_features = []  # Эти столбцы считаем не нужными и выкидываем
    numeric = ['Stuffs', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
               'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
               'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
               'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'BsmtFinSF2', 'BsmtFinSF1', 'SalePrice', 'MSSubClass',
               'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'LotFrontage',
               'MasVnrArea']
    categorical = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual',
                   'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'Exterior1st', 'Exterior2nd',
                   'ExterQual', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish',
                   'GarageQual', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour',
                   'LandSlope', 'LotConfig', 'LotShape', 'MasVnrType', 'MiscFeature', 'MSZoning', 'Neighborhood',
                   'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'Utilities']
    categorical_NA = ['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Fence',
                      'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'MiscFeature', 'PoolQC']
    categorical_mode = ['BldgType', 'CentralAir', 'Condition1', 'Condition2', 'Electrical', 'ExterCond', 'Exterior1st',
                        'Exterior2nd', 'ExterQual', 'Foundation', 'Functional', 'Heating', 'HeatingQC', 'HouseStyle',
                        'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MasVnrType', 'MSZoning',
                        'Neighborhood', 'PavedDrive', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street',
                        'Utilities']

    # Обработка цифровых данных
    # *************************************** WARNING ******************************
    numerical_columns = X.select_dtypes(exclude='object').columns
    num_features = X.select_dtypes(exclude='object')
    # *************************************** WARNING ******************************

    # Обратное разбиение на обучающую и тестовую выборки
    X_train = X.iloc[:X_train.shape[0], :]
    X_text = X.iloc[X_train.shape[0]:, :]


    def fill_with_min_pd(df):
        min_vals = df.min(axis=0)
        filled_df = df.fillna(min_vals)
        return filled_df

    # Выделение цифровых фич для замены пустых значений на медиану или на нули
    numeric_median_fill = ['LotFrontage', 'MasVnrArea']
    numeric_na_fill = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                       'GarageCars', 'GarageArea']
    # Обработка пропущенных значений в цифровых фичах
    num_imputer = ColumnTransformer(
        transformers=[
            ('numeric_median_fill', SimpleImputer(strategy='median'), numeric_median_fill),
            ('garage_imputer', FunctionTransformer(fill_with_min_pd, validate=False), ['GarageYrBlt']),
            ('numeric_na_fill', SimpleImputer(strategy='constant', fill_value=0), numeric_na_fill)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    filled_num_data = num_imputer.fit_transform(X_train, y)

    # Замена пропусков в категориальных переменных на моду, либо на строку 'Nothing'
    my_imputer = ColumnTransformer(
        transformers = [
            ('drop_features', 'drop', drop_features),
            ('cat_imputer_mode', SimpleImputer(strategy='most_frequent'), categorical_mode),
            ('cat_imputer_NA', SimpleImputer(strategy='constant', fill_value='Nothing'), categorical_NA)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    filled_data = my_imputer.fit_transform(filled_num_data, y)

    # Сортируем категориальные фичи под разные мтоды кодирования:
    # ранговые - методом OrdinalEncoding, разнородные - методом TargetEncoding или OHE
    ordinal_encoding_columns = ['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual',
                                'CentralAir', 'ExterCond', 'ExterQual', 'Fence', 'FireplaceQu', 'Functional',
                                'GarageCond', 'GarageFinish', 'GarageQual', 'HeatingQC', 'KitchenQual', 'LandSlope',
                                'PavedDrive', 'PoolQC', 'Street', 'Utilities']
    target_encoding_columns = ['BldgType', 'Condition1', 'Condition2', 'Electrical', 'Exterior1st', 'Exterior2nd',
                               'Foundation', 'GarageType', 'Heating', 'HouseStyle', 'LandContour', 'LotConfig',
                               'LotShape', 'MasVnrType', 'MiscFeature', 'MSZoning', 'Neighborhood', 'RoofMatl',
                               'RoofStyle', 'SaleCondition', 'SaleType']
    # *************************************** WARNING ******************************
    standard_scaler_columns = numerical_columns  # Числовые столбцы, которые необходимо пронормировать, теперь одна шкала

    # Создаем конвеер для кодирования категориальных фич
    my_encoder_scaler = ColumnTransformer(
        [
            ('ordinal_encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
             ordinal_encoding_columns),
            ('target_encoding', TargetEncoder(), target_encoding_columns),
            ('scaling_num_columns', StandardScaler(), standard_scaler_columns),
        ],
        verbose_feature_names_out=False,
        force_int_remainder_cols=False,
        remainder='passthrough'
    )
    encoded_data = my_encoder_scaler.fit_transform(filled_data, y)

    my_scaler = ColumnTransformer(
        [
            ('scaling', StandardScaler(), categorical)
        ],
        verbose_feature_names_out=False,
        force_int_remainder_cols=False,
        remainder='passthrough'
    )
    scaled_data = my_scaler.fit_transform(encoded_data)

    my_preprocessor = Pipeline(
        [
            ('num_imputer', num_imputer),
            ('cat_imputer', my_imputer),
            ('cat_encoder_num_scaler', my_encoder_scaler),
            ('cat_scaler', my_scaler)
        ]
    )

    rf = RandomForestRegressor(n_estimators=30, random_state=1)

    full_pipeline = Pipeline(
        [
            ('preprocessor', my_preprocessor),  # The preprocessing steps
            ('model', rf)  # The Random Forest model
        ]
    )

    # Define RLMSE function
    def rlmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # Random Forest
    def rf_objective(trial):
        # Параметры самой модели
        model_params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10, 1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200, 50),
        }

        full_pipeline.named_steps['model'].set_params(**model_params)

        # Создание сплитов для кросс-валидации
        cv = KFold(n_splits=5, random_state=666, shuffle=True)

        log_y = np.log(y)

        # Вычисление метрик точности с использованием кросс-валидации
        y_pred = cross_val_predict(full_pipeline, X_train, log_y, cv=cv)

        rlmse_score = rlmse(log_y, y_pred)

        return rlmse_score

    # Подбор лучших параметров с помощью OPTUNA
    # study_rf = optuna.create_study(direction='minimize')
    # study_rf.optimize(rf_objective, n_trials=100)

    # best_params_rf = study_rf.best_params
    # best_value_rf = study_rf.best_value

    # УДАЛИТЬ!!!
    # Optuna дала n_estimators = 100, max_depth = 10 для алгоритма RandomForestRegressor
    # best_params_rf = {'max_depth': 10, 'n_estimators': 100}
    # best_value_rf = 0.14174062146926147

    # Combine the preprocessor and the model into one pipeline и применяем лучшие параметры от OPTUNA
    rf_final = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=10)

    final_pipeline = Pipeline(
        [
            ('preprocessor', my_preprocessor),  # The preprocessing steps
            ('model', rf_final)  # The Random Forest model
        ]
    )

    final_pipeline.fit(X_train, np.log(y))

    log_predict = final_pipeline.predict(X_test)
    actual_values = np.exp(log_predict)
    ids = np.arange(1461, 2920)
    our_submission = pd.DataFrame({
        'Id': ids,
        'SalePrice': actual_values
    })

    st.write('Результат анализа данные - итоговые цены на недвижимость')
    st.write(our_submission)

    # Выгрузка данных анализа в CSV файл
    download_button = st.download_button(
        label='Скачать заполненный датасет в .csv',
        data=our_submission.to_csv(index=False),
        file_name='submission.csv')

else:
    st.stop()


