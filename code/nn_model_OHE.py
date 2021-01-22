import sys
sys.path.append('./')
import pandas as pd
import numpy as np
import glob
import typing
from config import mapping
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import utils
from torch.utils import data
import torch
import model
import pickle
import collections
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def assign_unique_id(player_data):
    player_data = player_data.assign(unique_id = lambda x : x['Tm'] + '_' + x['Opp'],
                                     Home = lambda x : x['Tm'], Away = lambda x : x['Opp'])
    mask = ~(player_data['at'].isna())
    player_data.loc[mask, 'unique_id'] = player_data.loc[mask, 'Opp'] + '_' + player_data.loc[mask, 'Tm']
    player_data.loc[mask, 'Home'] = player_data.loc[mask, 'Opp']
    player_data.loc[mask, 'Away'] = player_data.loc[mask, 'Home']
    return player_data

def encode_categorical_data(df):
    categorical_features = ['Home', 'Away', 'unique_id', 'Location', 'month', 'Season']
    df['month'] = pd.to_datetime(df['Date']).dt.month
    for feature in categorical_features:
        df.loc[:, feature + '_cat'] = df[feature].astype('category').cat.codes
    categorical = [c + '_cat' for c in categorical_features]
    return df, categorical

class SklearnWrapper:
    def __init__(self, transformation: typing.Callable):
        self.transformation = transformation
        self._group_transforms = []
        # Start with -1 and for each group up the pointer by one
        self._pointer = -1

    def _call_with_function(self, df: pd.DataFrame, function: str):
        # If pointer >= len we are making a new apply, reset _pointer
        if self._pointer >= len(self._group_transforms):
            self._pointer = -1
        self._pointer += 1
        return pd.DataFrame(
            getattr(self._group_transforms[self._pointer], function)(df.values),
            columns=df.columns,
            index=df.index,
        )

    def fit(self, df):
        self._group_transforms.append(self.transformation.fit(df.values))
        return self

    def transform(self, df):
        return self._call_with_function(df, "transform")

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df):
        return self._call_with_function(df, "inverse_transform")

def normalize_data(df_train, df_valid, df_test, feature):
    scaler = StandardScaler()
    df_train.loc[:, feature] = scaler.fit_transform(df_train[feature].values)
    df_valid.loc[:, feature] = scaler.fit_transform(df_valid[feature].values)
    df_test.loc[:, feature] = scaler.fit_transform(df_test[feature].values)
    return df_train, df_valid, df_test


if __name__ == '__main__':
    process = False
    dir = '../data/Basketball/Team/gamelog/'
    odds_data_path = '../data/scraped_odds_data.csv'

    teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET',
             'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
             'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS',
             'TOR', 'UTA', 'WAS']

    # first_year = 1985
    # last_year = 2019
    #
    # df = utils.process_data_all(dir, teams, first_year, last_year)
    #
    # df = df.sort_values(by = ['Date']).reset_index(drop = True)
    # df = df.drop_duplicates(['unique_id', 'Date'])
    # df = df.fillna(0.0)
    # Note : gameplay_data.csv already uses ex-ante features
    # train_nn = True
    df = pd.read_csv('../data/Basketball/gameplay_data.csv', header = 0)
    shift_num_games = 5
    features = df.columns[df.columns.str.contains('home') | df.columns.str.contains('away')].tolist()

    for feat in features:
        df.loc[:, feat + '_rolling_avg_100'] = df.groupby(['Home'])[feat].transform(lambda x : x.rolling(100).mean())
        if 'c_' not in feat:
            for i in range(1, shift_num_games + 1):
                df.loc[:, feat + '_last_{}_game_played'.format(i)] = df.groupby(['unique_id'])[feat].transform(lambda x : x.shift(i))

    numeric_features = features + df.columns[df.columns.str.contains('rolling_avg_100') |
                                     df.columns.str.contains('_game_played')].tolist()

    print('shape of data before dropping is {}'.format(df.shape))
    df = df.dropna(subset = numeric_features, axis = 0)
    print('shape of data after dropping is {}'.format(df.shape))

    df, categorical_features = encode_categorical_data(df)

    # cat_szs = [(df[c].nunique() + 1, min(50, (df[c].nunique())//2 )) for c in categorical_features]
    cat_szs = [(df[c].nunique(), df[c].nunique()) for c in categorical_features]
    df = pd.get_dummies(df, columns=categorical_features)

    cat_features = []
    for i in categorical_features:
        cat_features += df.columns[df.columns.str.contains(i)].tolist()

    all_features = features + cat_features
    print('Length of Features Is {}'.format(len(features)))


    # X_train = df.loc[(df['Season'] < 2008), all_features].values
    # y_train = df.loc[(df['Season'] < 2008), 'target'].values
    #
    # X_valid = df.loc[(df['Season'] >= 2008) & (df['Season'] < 2010), all_features].values
    # y_valid = df.loc[(df['Season'] >= 2008) & (df['Season'] < 2010), 'target'].values
    #
    # X_test = df.loc[(df['Season'] >= 2010), all_features].values
    # y_test = df.loc[(df['Season'] >= 2010), 'target'].values

    X_train_df = df.loc[(df['Season'] < 2008), :].reset_index(drop = True)
    y_train_df = df.loc[(df['Season'] < 2008), 'target'].reset_index(drop = True)

    X_valid_df = df.loc[(df['Season'] >= 2008) & (df['Season'] < 2010), :].reset_index(drop = True)
    y_valid_df = df.loc[(df['Season'] >= 2008) & (df['Season'] < 2010), 'target'].reset_index(drop = True)

    X_test_df = df.loc[(df['Season'] >= 2010), :].reset_index(drop = True)
    y_test_df = df.loc[(df['Season'] >= 2010), 'target'].reset_index(drop = True)


    X_train_df, X_valid_df, X_test_df = normalize_data(X_train_df, X_valid_df, X_test_df, numeric_features)

    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 6}

    cont_train_tensor = torch.from_numpy(X_train_df[numeric_features].values).float()
    cat_train_tensor = torch.from_numpy(X_train_df[cat_features].values.astype('int')).float()

    cont_valid_tensor = torch.from_numpy(X_valid_df[numeric_features].values).float()
    cat_valid_tensor = torch.from_numpy(X_valid_df[cat_features].values.astype('int')).float()

    train_label_tensor = torch.from_numpy(y_train_df.values.astype('float')).float().view(-1,1)
    valid_label_tensor = torch.from_numpy(y_valid_df.values.astype('float')).float().view(-1,1)
    #
    print(train_label_tensor.size())

    train_dataset = torch.utils.data.TensorDataset(cat_train_tensor,
                                             cont_train_tensor,
                                             train_label_tensor)

    valid_dataset = torch.utils.data.TensorDataset(cat_valid_tensor,
                                                   cont_valid_tensor,
                                                   valid_label_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 256, shuffle = True, drop_last=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle = True,
                                                   drop_last= True, pin_memory=True)

    args = {
        'emb_dims' : cat_szs,
        'continuous_features' : len(numeric_features),
        'lin_layer_sizes' : [512, 256],
        'output_size' : 1,
        'embedding_dropout' : 0.2,
        'linear_layer_dropout' : [0.5, 0.5],
        'lr' : 1e-4,
        'use_embedding' : False,
        'log_dir' : '../output/neural_network_training/ohe/'
    }

    epochs = model.train(args, patience = 25, train_dataset = train_loader, valid_dataset = valid_loader)
    #
    dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle = True, drop_last=True, pin_memory=True)
    nn_model, loss = model.train(args, patience = 25, num_epochs = epochs, early_stopping = False, dataset = dataloader)

    cont_test_tensor = torch.from_numpy(X_test_df[numeric_features].values).float()
    cat_test_tensor = torch.from_numpy(X_test_df[cat_features].values.astype('int')).float()
    test_label_tensor = torch.from_numpy(y_test_df.values.astype('float')).float().view(-1,1)

    test_dataset = torch.utils.data.TensorDataset(cat_test_tensor,
                                                  cont_test_tensor,
                                                  test_label_tensor)

    test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle = False, drop_last=False,
                                             pin_memory=True)
    train_pred_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle = False, drop_last=False, pin_memory=True)
    valid_pred_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False, drop_last=False, pin_memory=True)

    y_test_pred, log_preds_test = model.make_predictions(test_loader, nn_model)
    y_train_pred, log_preds_train = model.make_predictions(train_pred_loader, nn_model)
    y_valid_pred, log_preds_valid = model.make_predictions(valid_pred_loader, nn_model)

    y_pred_binary = (y_test_pred > 0.5)
    y_pred_train_binary = (y_train_pred > 0.5)
    y_pred_valid_binary = (y_valid_pred > 0.5)

    nn_train_acc = accuracy_score(y_train_df.values, y_pred_train_binary)
    nn_valid_acc = accuracy_score(y_valid_df.values, y_pred_valid_binary)
    nn_test_acc = accuracy_score(y_test_df.values, y_pred_binary)

    print('Train Accuracy is {}'.format(nn_train_acc))
    print('Valid Accuracy is {}'.format(nn_valid_acc))
    print('Test Accuracy is {}'.format(nn_test_acc))
    #
    profit, nn_cum_profit = utils.backtest(y_pred_binary, y_test_pred, 'nn_model_ohe')

    # train gradient boosting
    # params_1 = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'learning_rate': 0.01,
    #     'verbosity': 0
    # }
    #
    # lgb_features = features + categorical_features
    #
    # train_data = lgb.Dataset(X_train_df[lgb_features].values, label=y_train_df.values.astype(int))
    # valid_data = lgb.Dataset(X_valid_df[lgb_features].values, label=y_valid_df.values.astype(int))
    # test_data = lgb.Dataset(X_test_df[lgb_features].values, label = y_test_df.values.astype(int))
    #
    # gbm_1 = lgb.train(params_1,
    #                   train_data,
    #                   num_boost_round=2000,
    #                   valid_sets=valid_data, verbose_eval=100,
    #                   early_stopping_rounds=500)
    #
    # y_pred_test = gbm_1.predict(X_test_df[lgb_features].values)
    # lgb_ypred_binary = (y_pred_test > 0.5)
    # lgb_ypred_train = (gbm_1.predict(X_train_df[lgb_features]) > 0.5)
    # lgb_ypred_valid = (gbm_1.predict(X_valid_df[lgb_features]) > 0.5)
    #
    # lgb_train_acc = accuracy_score(y_train_df.values, lgb_ypred_train)
    # lgb_valid_acc = accuracy_score(y_valid_df.values, lgb_ypred_valid)
    # lgb_test_acc = accuracy_score(y_test_df.values, lgb_ypred_binary)
    #
    # print('Train Accuracy is {}'.format(lgb_train_acc))
    # print('Valid Accuracy is {}'.format(lgb_valid_acc))
    # print('Test Accuracy is {}'.format(lgb_test_acc))
    #
    # lgb_profit, lgb_cum_profit = utils.backtest(lgb_ypred_binary, y_pred_test, 'lgb')
    #
    # qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    # qda_model = qda.fit(X_train_df[features], y_train_df.values)
    # qda_valid_predict = qda.predict(X_valid_df[features])
    # qda_train_predict = qda.predict(X_train_df[features])
    # qda_test_predict = qda.predict(X_test_df[features])
    #
    # qda_train_prob = qda.predict_proba(X_train_df[features])
    # qda_valid_prob = qda.predict_proba(X_valid_df[features])
    # qda_test_prob = qda.predict_proba(X_test_df[features])
    #
    # qda_train_acc = accuracy_score(y_train_df.values, qda_train_predict)
    # qda_valid_acc = accuracy_score(y_valid_df.values, qda_valid_predict)
    # qda_test_acc = accuracy_score(y_test_df.values, qda_test_predict)
    #
    # print('Train Accuracy is {}'.format(qda_train_acc))
    # print('Valid Accuracy is {}'.format(qda_valid_acc))
    # print('Test Accuracy is {}'.format(qda_test_acc))
    #
    # qda_profit, qda_cum_profit = utils.backtest(qda_test_predict, qda_test_prob[:,0], 'qda')

    # lda = LinearDiscriminantAnalysis(store_covariance=True)
    # lda_model = lda.fit(X_train_df[features], y_train_df.values)
    # lda_valid_predict = lda.predict(X_valid_df[features])
    # lda_train_predict = lda.predict(X_train_df[features])
    # lda_test_predict = lda.predict(X_test_df[features])
    #
    # lda_train_prob = lda.predict_proba(X_train_df[features])
    # lda_valid_prob = lda.predict_proba(X_valid_df[features])
    # lda_test_prob = lda.predict_proba(X_test_df[features])
    #
    # lda_train_acc = accuracy_score(y_train_df.values, lda_train_predict)
    # lda_valid_acc = accuracy_score(y_valid_df.values, lda_valid_predict)
    # lda_test_acc = accuracy_score(y_test_df.values, lda_test_predict)
    #
    # print('Train Accuracy is {}'.format(lda_train_acc))
    # print('Valid Accuracy is {}'.format(lda_valid_acc))
    # print('Test Accuracy is {}'.format(lda_test_acc))

    # qda_profit, qda_cum_profit = utils.backtest(lda_test_predict, lda_test_prob[:,0], 'qda')
    # svc = SVC(gamma = 'scale', probability=True)
    # svc_model = svc.fit(X_train_df[features], y_train_df.values)
    # svc_train_predict = svc_model.predict(X_train_df[features])
    # svc_valid_predict = svc_model.predict(X_valid_df[features])
    # svc_test_predict = svc_model.predict(X_test_df[features])
    #
    # svc_train_prob = svc_model.predict_proba(X_train_df[features])
    # svc_valid_prob = svc_model.predict_proba(X_valid_df[features])
    # svc_test_prob = svc_model.predict_proba(X_test_df[features])
    #
    # svc_train_acc = accuracy_score(y_train_df.values, svc_train_predict)
    # svc_valid_acc = accuracy_score(y_valid_df.values, svc_valid_predict)
    # svc_test_acc = accuracy_score(y_test_df.values, svc_test_predict)
    #
    # print('Train Accuracy is {}'.format(svc_train_acc))
    # print('Valid Accuracy is {}'.format(svc_valid_acc))
    # print('Test Accuracy is {}'.format(svc_test_acc))
    #
    # svc_profit, svc_cum_profit = utils.backtest(svc_test_predict, svc_test_prob[:,1], 'svc')
    #
    # accuracy = pd.DataFrame({'train' : [0.6795, 0.7024, 0.6563, 0.6585 ,0.7971],
    #                          'valid' : [0.6701, 0.6263, 0.4119, 0.5025 ,0.6222],
    #                          'test' : [0.6232, 0.6114, 0.4334,  0.4723 ,0.6097]}, index=['NN', 'gradient_boosting', 'qda', 'lda', 'svc'])
    # accuracy = accuracy.stack()
    # accuracy = accuracy.reset_index(drop = False)
    # accuracy.rename({'level_1' : 'sets', 0:'Acc', 'level_0': 'Models'}, axis = 1, inplace = True)
    # # accuracy = accuracy.reset_index(drop = False)
    # fig, ax = plt.subplots(figsize = (8,8))
    # ax = sns.barplot(x = 'sets', y = 'Acc', hue = 'Models', ax = ax, data = accuracy)
    # ax.set(title = 'Model Accuracy Scores For Train/Valid/Test')
    # plt.savefig('../output/accuracy_score.jpg', format = 'jpg')
    # plt.show()
    #
    # player_df = pd.read_csv('../data/Basketball/players_data.csv', header = 0, index_col = 0)
    #
    # cols = ['+/-', '3P', '3P%', '3PA', 'AST', 'BLK', 'DRB','FG', 'FG%', 'FGA', 'FT', 'FT%',
    #         'FTA', 'PTS', 'STL', 'TOV', 'TRB']
    # player_df['Date'] = pd.to_datetime(player_df['Date'])
    # player_df = player_df.sort_values(['Date', 'name', 'Tm'])
    # grouped_df = player_df.groupby(['name'])
    #
    # for c in cols:
    #     player_df.loc[:, c] = grouped_df[c].transform(lambda x : x.shift(1))
    #
    # grouped_df = player_df.groupby(['name'])
    # for c in cols:
    #     try:
    #         player_df.loc[:, c + '_avg'] = grouped_df[c].transform(lambda x : x.rolling(50, min_periods=10).mean())
    #     except:
    #         print('Error on {}'.format(c))
    #
    # avg_cols = [c + '_avg' for c in cols]
    # threshold = {'+/-' : 8, '3PA' : 5, 'AST' : 7, 'FG' : 7, 'FG%' : 50, 'FT' : 5,
    #              'FT%' : 80, 'FTA' : 5, 'PTS' : 20, 'STL' : 3, 'TOV' : 4, 'TRB' : 4,
    #              '3P' : 3,
    #              '3P%': 50}
    # threshold_avg = {}
    # for k, v in threshold.items():
    #     threshold_avg[k + '_avg'] = v
    #
    # grouped_df = player_df.groupby(['Date', 'Tm'])
    # for k, v in threshold.items():
    #     player_df.loc[:, k + '_counts'] = grouped_df[avg_cols].apply(lambda x : (x > v).sum())
    #
    # methods = {}
    # for k, v in threshold.items():
    #     methods[k] = 'mean'
    #
    # df_agg = player_df.groupby(['Date', 'Tm']).agg(methods)
    #
