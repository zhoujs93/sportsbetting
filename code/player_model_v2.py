import argparse
import sys
sys.path.append('./')
import pandas as pd
import numpy as np
import typing
from sklearn.preprocessing import StandardScaler
import utils
from torch.utils import data
import torch
import model
import pathlib
from webscraper import load_obj, save_obj

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

def aggregate_player_data():

    player_df = pd.read_csv('../data/Basketball/players_data.csv', header = 0, index_col = 0)

    cols = ['+/-', '3P', '3P%', '3PA', 'AST', 'BLK', 'DRB','FG', 'FG%', 'FGA', 'FT', 'FT%',
            'FTA', 'PTS', 'STL', 'TOV', 'TRB']
    player_df['Date'] = pd.to_datetime(player_df['Date'])
    player_df = player_df.sort_values(['Date', 'name', 'Tm'])
    grouped_df = player_df.groupby(['name'])

    for c in cols:
        player_df.loc[:, c] = grouped_df[c].transform(lambda x : x.shift(1))

    grouped_df = player_df.groupby(['name'])
    for c in cols:
        try:
            player_df.loc[:, c + '_avg'] = grouped_df[c].transform(lambda x : x.rolling(50, min_periods=10).mean())
        except:
            print('Error on {}'.format(c))

    avg_cols = [c + '_avg' for c in cols]
    threshold = {'+/-' : 8, '3PA' : 5, 'AST' : 7, 'FG' : 7, 'FG%' : 50, 'FT' : 5,
                 'FT%' : 80, 'FTA' : 5, 'PTS' : 20, 'STL' : 3, 'TOV' : 4, 'TRB' : 4,
                 '3P' : 3,
                 '3P%': 50}
    threshold_avg = {}
    for k, v in threshold.items():
        threshold_avg[k + '_avg'] = v

    grouped_df = player_df.groupby(['Date', 'Tm'])
    for k, v in threshold.items():
        player_df.loc[:, k + '_counts'] = grouped_df[avg_cols].apply(lambda x : (x > v).sum())

    methods = {}
    for k, v in threshold.items():
        methods[k] = 'mean'

    df_agg = player_df.groupby(['Date', 'Tm']).agg(methods)

    return df_agg

def normalize_data(df_train, df_valid, df_test, feature):
    scaler = StandardScaler()
    df_train.loc[:, feature] = scaler.fit_transform(df_train[feature].values)
    df_valid.loc[:, feature] = scaler.fit_transform(df_valid[feature].values)
    df_test.loc[:, feature] = scaler.fit_transform(df_test[feature].values)
    return df_train, df_valid, df_test

def generate_test_predictions(df, args):
    home = ['LAL']
    vals = {'Home' : ['LAL'],
            'Away' : ['MIA'],
            'Opp' : ['LAL'],
            'Date': [args.date] * len(home),
            'unique_id' : ['LAL_MIA'],
            'Season' : [2020] * len(home)}
    df_vals = pd.DataFrame(vals)
    df_vals_a = df.columns.tolist()
    c = df_vals.columns[~df_vals.columns.isin(df_vals_a)].tolist()
    df_vals[c] = np.NaN
    tt = pd.concat([df, df_vals], axis=0, ignore_index=True)
    home_feats = df.columns[df.columns.str.contains('home')]
    away_feats = df.columns[df.columns.str.contains('away')]
    for home_feat, away_feat in zip(home_feats, away_feats):
        tt.loc[:, home_feat] = tt.groupby(['Home'])[home_feat].transform(lambda x: x.fillna(method = 'ffill'))
        tt.loc[:, away_feat] = tt.groupby(['Away'])[away_feat].transform(lambda x: x.fillna(method = 'ffill'))
    return tt

def arg_parse():
    parser = argparse.ArgumentParser(description = 'Player Model')
    parser.add_argument('-production', default = '0', type = int, help = 'production or train')
    parser.add_argument('-date', type = str, default='2020-10-06', help = 'prediction date')
    parser.add_argument('-model', type = str, default = 'regression', help = 'options: moneyline vs total')
    parser.add_argument('-threshold', type = int, default = '50', help = 'only used if model = total')
    parser.add_argument('-load_processed_data', type = int, default = 1, help = 'load preprocessed data')
    parser.add_argument('-save_processed_data', type = int, default = 0, help = 'save preprocessed data')
    parser.add_argument('-lin_layer_size', nargs='+', type=int, default = [512, 256], help = 'linear layers')
    parser.add_argument('-lin_layer_dropout', nargs = '+', type = float, default = [0.5, 0.5], help = 'dropout for each lin layer')
    parser.add_argument('-log_dir_folder', default = 'regression_10_17', type = str, help = 'log_dir_folder')
    parser.add_argument('-early_stopping', default = 50, type = int, help = 'early_stopping')
    parser.add_argument('-swa', default = 0, type = int, help = 'swa')
    parser.add_argument('-lr', default = 5e-4, type = float, help = 'lr')
    parser.add_argument('-weighted_sampler', type = int, default = 0, help = 'weighted sampler')
    parser.add_argument('-scheduler', default = 'cosine', type = str, help = 'cosine_annealing vs stepLR')
    return parser


if __name__ == '__main__':
    arg = arg_parse()
    arg = arg.parse_args()

    print(f'Args are: {arg}')

    assert len(arg.lin_layer_size) == len(arg.lin_layer_dropout)

    process = False

    dir = '../data/Basketball/Team/gamelog/'
    odds_data_path = '../data/scraped_odds_data.csv'

    teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET',
             'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
             'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS',
             'TOR', 'UTA', 'WAS']
    #
    first_year = 1985
    last_year = 2020
    #
    if not arg.load_processed_data:
        df = utils.process_data_all(dir, arg)
        #
        df = df.sort_values(by = ['Date']).reset_index(drop = True)
        df = df.drop_duplicates(['unique_id', 'Date'])
        df = df.fillna(0.0)

        home_feats = df.columns[df.columns.str.contains('home')]
        away_feats = df.columns[df.columns.str.contains('away')]
        for home_feat, away_feat in zip(home_feats, away_feats):
            df.loc[:, home_feat] = df.groupby(['Home'])[home_feat].transform(lambda x: x.shift(1))
            df.loc[:, away_feat] = df.groupby(['Away'])[away_feat].transform(lambda x: x.shift(1))

        # Note : gameplay_data.csv already uses ex-ante features
        # train_nn = True
        # df_t = pd.read_csv('../data/Basketball/gameplay_data.csv', header = 0)
        df['Date'] = pd.to_datetime(df['Date'])
        shift_num_games = 5
        features = df.columns[df.columns.str.contains('home') | df.columns.str.contains('away')].tolist()
        # generate and process player data
        df_agg = aggregate_player_data()
        df_agg = df_agg.reset_index(drop = False)
        col_renamed = df_agg.columns[:2].tolist() + ['count_' + c for c in df_agg.columns[2:].tolist()]
        df_agg.columns = col_renamed
        df = df.merge(df_agg, how = 'left', left_on = ['Date', 'Home'], right_on = ['Date', 'Tm'])
        df = df.merge(df_agg, how = 'left', left_on = ['Date', 'Away'], right_on = ['Date','Tm'])
        count_cols = df.columns[df.columns.str.contains('count')].tolist()
        # mappings = {c : 0 for c in count_cols}
        for c in count_cols:
            df.loc[:, c] = df.groupby(['Home'])[c].transform(lambda x: x.fillna(method = 'ffill'))
            df.loc[:, c] = df.groupby(['Away'])[c].transform(lambda x: x.fillna(method='ffill'))

        df_copy = df.copy()
        # TODO: Need to modify this
        df = generate_test_predictions(df_copy, arg)

        for c in count_cols:
            df.loc[:, c] = df.groupby(['Home'])[c].transform(lambda x: x.fillna(method = 'ffill'))
            df.loc[:, c] = df.groupby(['Away'])[c].transform(lambda x: x.fillna(method='ffill'))

        all_features = []
        for feat in features:
            for period in [2, 3, 4, 5, 10, 20, 30, 40]:
                cur_feature = feat + f'_rolling_avg_{period}'
                all_features.append(cur_feature)
                df.loc[:, feat + f'_rolling_avg_{period}'] = df.groupby(['Home'])[feat].transform(lambda x : x.rolling(period).mean())
                if 'c_' not in feat:
                    for i in range(1, shift_num_games + 1):
                        df.loc[:, feat + '_last_{}_game_played'.format(i)] = df.groupby(['unique_id'])[feat].transform(lambda x : x.shift(i))

        if arg.save_processed_data:
            data_dir = pathlib.Path.cwd().parent / 'data'
            filename = str(data_dir) + '/player_model_features.feather'
            df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')
            df.to_feather(filename)
            all_feature_fname = str(data_dir) + '/all_features.pickle'
            save_obj(all_feature_fname, all_features)
            count_cols_fname = str(data_dir) + '/count_cols.pickle'
            save_obj(count_cols_fname, count_cols)
            features_fname = str(data_dir) + '/features.pickle'
            save_obj(features_fname, features)
    else:
        data_dir = pathlib.Path.cwd().parent / 'data'
        filename = str(data_dir) + '/player_model_features.feather'
        all_feature_fname = str(data_dir) + '/all_features.pickle'
        count_cols_fname = str(data_dir) + '/count_cols.pickle'
        features_fname = str(data_dir) + '/features.pickle'
        df = pd.read_feather(filename)
        all_features = load_obj(all_feature_fname)
        features = load_obj(features_fname)
        count_cols = load_obj(count_cols_fname)
        df = utils.generate_labels(df, arg)
        df = df.assign(year = lambda x: x['Date'].dt.year)
        all_features += ['year']


    cols = [i for i in df.columns if i in all_features]
    features = features + all_features + df.columns[df.columns.str.contains('_game_played')].tolist() + count_cols

    bias = 'score_diff' in features
    print(f'is score_diff in Features ? {bias}')

    df = df.reset_index(drop = True)
    df = df.fillna(0.0)
    print('shape of data before dropping is {}'.format(df.shape))
    df = df.dropna(subset = features, axis = 0)
    print('shape of data after dropping is {}'.format(df.shape))

    predict_copy = df.loc[df['Date'] == arg.date, :]

    df, categorical_features = encode_categorical_data(df)

    cat_szs = [(df[c].nunique() + 1, min(50, (df[c].nunique())//2 )) for c in categorical_features]

    predict = df.loc[df['Date'] == arg.date, :]
    prediction_date = arg.date
    predict_bool = True
    predictions = {}
    #
    for seed in range(1, 10):
        if predict_bool == False:
            X_train_df = df.loc[(df['Season'] < 2008), :].reset_index(drop = True)
            y_train_df = df.loc[(df['Season'] < 2008), 'target'].reset_index(drop = True)

            X_valid_df = df.loc[(df['Season'] >= 2008) & (df['Season'] < 2010), :].reset_index(drop = True)
            y_valid_df = df.loc[(df['Season'] >= 2008) & (df['Season'] < 2010), 'target'].reset_index(drop = True)

            X_test_df = df.loc[(df['Season'] >= 2010) & (df['Date'] != prediction_date), :].reset_index(drop = True)
            y_test_df = df.loc[(df['Season'] >= 2010) & (df['Date'] != prediction_date), 'target'].reset_index(drop = True)
        else:
            X_train_df = df.loc[(df['Season'] < 2020), :].reset_index(drop=True)
            y_train_df = df.loc[(df['Season'] < 2020), 'target'].reset_index(drop=True)

            X_valid_df = df.loc[(df['Season'] >= 2020), :].reset_index(drop=True)
            y_valid_df = df.loc[(df['Season'] >= 2020), 'target'].reset_index(drop=True)

            X_test_df = predict.reset_index(drop = True)
            y_test_df = predict.loc[:, 'target'].reset_index(drop=True)

        print(f'Proportion of 1 is for train set is : {y_train_df.sum() / y_train_df.shape[0]}')
        print(f'Proportion of 1 is for valid set is : {y_valid_df.sum() / y_valid_df.shape[0]}')
        print(f'Proportion of 1 is for test set is : {y_test_df.sum() / y_test_df.shape[0]}')


        pos_count = y_train_df.sum()
        neg_count = y_train_df.shape[0] - pos_count
        pos_weight = (neg_count / pos_count) * 1.5

        print(f'Pos weight is {pos_weight}')

        X_train_df, X_valid_df, X_test_df = normalize_data(X_train_df, X_valid_df, X_test_df, features)

        cont_train_tensor = torch.from_numpy(X_train_df[features].values).float()
        cat_train_tensor = torch.from_numpy(X_train_df[categorical_features].values.astype('int')).long()

        cont_valid_tensor = torch.from_numpy(X_valid_df[features].values).float()
        cat_valid_tensor = torch.from_numpy(X_valid_df[categorical_features].values.astype('int')).long()

        train_label_tensor = torch.from_numpy(y_train_df.values.astype('float')).float().view(-1,1)
        valid_label_tensor = torch.from_numpy(y_valid_df.values.astype('float')).float().view(-1,1)

        print(train_label_tensor.size())

        train_dataset = torch.utils.data.TensorDataset(cat_train_tensor,
                                                 cont_train_tensor,
                                                 train_label_tensor)

        valid_dataset = torch.utils.data.TensorDataset(cat_valid_tensor,
                                                       cont_valid_tensor,
                                                       valid_label_tensor)

        label_tr = y_train_df.to_frame('label').copy()
        label_tr = label_tr.assign(weights = 1)
        label_tr.loc[label_tr['label'] == 1, 'weights'] = pos_weight
        label_val = y_valid_df.to_frame('label').copy()
        label_val = label_val.assign(weights = 1)
        label_val.loc[label_val['label'] == 1, 'weights'] = 1
        all_wts = pd.concat([label_tr, label_val], axis = 0, ignore_index = True)
        all_wts.loc[all_wts['label'] == 1, 'weights'] = pos_weight


        if arg.model == 'total':
            log_dir = pathlib.Path.cwd().parent / 'output' / f'tb_{arg.log_dir_folder}' / f'{arg.date}_{arg.model}_{arg.threshold}_{seed}'
        else:
            log_dir = pathlib.Path.cwd().parent / 'output' / f'tb_{arg.log_dir_folder}' / f'{arg.date}_{arg.model}_{seed}'

        log_dir.mkdir(parents = True, exist_ok = True)

        args = {
            'emb_dims' : cat_szs,
            'continuous_features' : len(features),
            'lin_layer_sizes' : arg.lin_layer_size,
            'output_size' : 1,
            'embedding_dropout' : 0.2,
            'linear_layer_dropout' : arg.lin_layer_dropout,
            'lr' : arg.lr,
            'use_embedding' : True,
            'seed' : seed,
            'log_dir': str(log_dir),
            'objective' : arg.model,
            'swa' : arg.swa,
            'scheduler' : arg.scheduler
        }


        torch.manual_seed(args['seed'])
        np.random.seed(args['seed'])

        if arg.weighted_sampler:
            train_sampler = torch.utils.data.WeightedRandomSampler(label_tr['weights'].values.tolist(), label_tr.shape[0], replacement=True)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 256, sampler = train_sampler, drop_last=True, pin_memory=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle = True,
                                                           drop_last= True, pin_memory=True)
        else:

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True,
                                                       pin_memory=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=True,
                                                       drop_last=True, pin_memory=True)

        if arg.weighted_sampler:
            epochs = model.train(args, patience = arg.early_stopping, train_dataset = train_loader, valid_dataset = valid_loader,
                                 pos_weights = pos_weight)
        else:
            epochs = model.train(args, patience = arg.early_stopping, train_dataset = train_loader, valid_dataset = valid_loader)

        dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])

        if arg.weighted_sampler:
            all_sampler = torch.utils.data.WeightedRandomSampler(all_wts['weights'].values.tolist(), all_wts.shape[0],
                                                                 replacement=True)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, sampler = all_sampler, drop_last=True, pin_memory=True)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True,
                                                     pin_memory=True)



        if (pos_count / y_train_df.shape[0]) < 0.3:
            nn_model, loss = model.train(args, patience = arg.early_stopping, num_epochs = epochs, early_stopping = False,
                                         dataset = dataloader, pos_weights = pos_weight)
        else:
            nn_model, loss = model.train(args, patience=arg.early_stopping, num_epochs=epochs, early_stopping=False, dataset=dataloader)

        cont_test_tensor = torch.from_numpy(X_test_df[features].values).float()
        cat_test_tensor = torch.from_numpy(X_test_df[categorical_features].values.astype('int')).long()
        test_label_tensor = torch.from_numpy(y_test_df.values.astype('float')).float().view(-1,1)

        test_dataset = torch.utils.data.TensorDataset(cat_test_tensor,
                                                      cont_test_tensor,
                                                      test_label_tensor)

        test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle = False, drop_last=False,
                                                 pin_memory=True)
        train_pred_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle = False, drop_last=False, pin_memory=True)
        valid_pred_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False, drop_last=False, pin_memory=True)

        y_test_pred, log_preds_test = model.make_predictions(test_loader, nn_model, args)
        y_train_pred, log_preds_train = model.make_predictions(train_pred_loader, nn_model, args)
        y_valid_pred, log_preds_valid = model.make_predictions(valid_pred_loader, nn_model, args)

        y_pred_binary = (y_test_pred > 0.5)
        y_pred_train_binary = (y_train_pred > 0.5)
        y_pred_valid_binary = (y_valid_pred > 0.5)
        predictions[seed] = y_test_pred

    # predictions[232425] = np.array([0.30839, 0.63836, 0.74958, 0.73305]).reshape(-1,1)
    # predictions[111111] = np.array([0.22069, 0.59224, 0.64530, 0.75158]).reshape(-1,1)
    # ppp = {k : v.ravel() for k,v in predictions.items()}
    # pr = pd.DataFrame(ppp, index = list(range(4)))
    # probb = pr.mean(axis = 1)
    #
    # # stop here -- Jan 20th, 2020
    #
    # X_train_df = pd.read_feather('../output/nn_predictions/train_df.feather')
    # X_valid_df = pd.read_feather('../output/nn_predictions/valid_df.feather')
    # X_test_df = pd.read_feather('../output/nn_predictions/test_df.feather')
    #
    # y_pred_binary = X_test_df['nn_predictions_binary'].values
    # y_test_pred = X_test_df['nn_predictions_prob'].values
    # y_train_pred = X_train_df['nn_predictions_prob'].values
    # y_pred_train_binary = X_train_df['nn_predictions_binary'].values
    # y_valid_pred = X_valid_df['nn_predictions_prob'].values
    # y_pred_valid_binary = X_valid_df['nn_predictions_binary'].values
    #
    # nn_train_acc = accuracy_score(y_train_df.values, y_pred_train_binary)
    # nn_valid_acc = accuracy_score(y_valid_df.values, y_pred_valid_binary)
    # nn_test_acc = accuracy_score(y_test_df.values, y_pred_binary)
    #
    # print('Train Accuracy is {} ; Error is {}'.format(nn_train_acc, 1-nn_train_acc))
    # print('Valid Accuracy is {} ; Error is {}'.format(nn_valid_acc, 1-nn_valid_acc))
    # print('Test Accuracy is {}  ; Error is {}'.format(nn_test_acc, 1-nn_test_acc))
    #
    # profit, nn_cum_profit = utils.backtest(y_pred_binary, y_test_pred, 'nn_model_player')
    #
    # X_train_df['nn_predictions_prob'] = y_train_pred
    # X_train_df['nn_predictions_binary'] = y_pred_train_binary
    # X_valid_df['nn_predictions_prob'] = y_valid_pred
    # X_valid_df['nn_predictions_binary'] = y_pred_valid_binary
    # X_test_df['nn_predictions_prob'] = y_test_pred
    # X_test_df['nn_predictions_binary'] = y_pred_binary
    #
    # # train gradient boosting
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
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # feature_imp = pd.DataFrame(sorted(zip(gbm_1.feature_importance(), lgb_features)), columns = ['Value', 'Feature'])
    # plt.figure(figsize=(20, 10))
    # sns.barplot(x="Value", y="Feature", data=feature_imp.iloc[-20:,:].sort_values(by="Value", ascending=False))
    # plt.title('LightGBM Feature Importance Based on Split')
    # plt.tight_layout()
    # plt.savefig('../output/lgbm_importances_split.jpg')
    # plt.show()
    #
    # #
    # X_train_df, X_valid_df, X_test_df = normalize_data(X_train_df, X_valid_df, X_test_df, features)
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
    # qda_profit, qda_cum_profit = utils.backtest(qda_test_predict, qda_test_prob[:,1], 'qda')
    #
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
    #
    # lda_profit, lda_cum_profit = utils.backtest(lda_test_predict, lda_test_prob[:,1], 'lda')
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
    #
    # accuracy = pd.DataFrame({'train' : [nn_train_acc, lgb_train_acc, qda_train_acc, lda_train_acc ,svc_train_acc],
    #                          'valid' : [nn_valid_acc, lgb_valid_acc, qda_valid_acc, lda_valid_acc ,svc_valid_acc],
    #                          'test' : [nn_test_acc, lgb_test_acc, qda_test_acc,  lda_test_acc ,svc_test_acc]}, index=['NN', 'gradient_boosting', 'qda', 'lda', 'svc'])
    # accuracy = accuracy.stack()
    # accuracy = accuracy.reset_index(drop = False)
    # accuracy.rename({'level_1' : 'sets', 0:'Acc', 'level_0': 'Models'}, axis = 1, inplace = True)
    # # accuracy = accuracy.reset_index(drop = False)
    # fig, ax = plt.subplots(figsize = (8,8))
    # ax = sns.barplot(x = 'sets', y = 'Acc', hue = 'Models', ax = ax, data = accuracy)
    # ax.set(title = 'Model Accuracy Scores For Train/Valid/Test')
    # plt.savefig('../output/accuracy_score.jpg', format = 'jpg')
    # plt.show()
