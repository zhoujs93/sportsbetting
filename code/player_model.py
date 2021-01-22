import argparse
import sys

sys.path.append('./')
import pandas as pd
import numpy as np
import glob
import typing
from config import mapping, mapping_abbrev_full
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import utils
from torch.utils import data
import torch
import model
import pathlib
from webscraper import load_obj, save_obj


def assign_unique_id(player_data):
    player_data = player_data.assign(unique_id=lambda x: x['Tm'] + '_' + x['Opp'],
                                     Home=lambda x: x['Tm'], Away=lambda x: x['Opp'])
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


def aggregate_player_data_individual():
    player_df = pd.read_csv('../data/Basketball/players_data.csv', header=0, index_col=0)
    unnamed_cols = player_df.columns[player_df.columns.str.startswith('Unnamed')].tolist()
    cols_to_keep = player_df.columns[~player_df.columns.isin(unnamed_cols)]
    player_df = player_df[cols_to_keep].reset_index(drop = True)
    cols = ['+/-', '3P', '3P%', '3PA', 'AST', 'BLK', 'DRB', 'FG', 'FG%', 'FGA', 'FT', 'FT%',
            'FTA', 'PTS', 'STL', 'TOV', 'TRB']

    grouped_df = player_df.groupby(['name'])
    for c in cols:
        player_df.loc[:, c] = grouped_df[c].transform(lambda x: x.shift(1))

    player_df['Date'] = pd.to_datetime(player_df['Date'])
    player_df.loc[~player_df['MP'].isna(), 'MP'] = player_df.loc[~player_df['MP'].isna(), 'MP'].astype(str).apply(lambda x: x.replace(':', '.'))
    player_df['MP'] = player_df['MP'].apply(float)
    nan_idx = player_df['MP'].isna()
    player_df.loc[nan_idx, 'MP'] = (player_df.loc[nan_idx, :]
                                             .groupby(['Date'])['MP']
                                             .transform(lambda x: x.fillna(0.0) + 1e-4 * np.random.normal(0, 2, size = len(x))))
    player_df = player_df.sort_values(['Date', 'name', 'Tm'], ascending = True).reset_index(drop = True)
    player_df.loc[:, 'rank'] = player_df.groupby(['Date', 'Tm'])['MP'].transform(lambda x: x.rank(ascending = False, method = 'first'))
    player_df = player_df.assign(player_rank=lambda x: 'player_' + x['rank'].astype(str))
    player_df = player_df.loc[player_df['rank'] <= 7, :].reset_index(drop = True)
    test = player_df.loc[player_df['Tm'] == 'LAL', :]
    df_player = pd.pivot_table(player_df, values = cols, index=['Date', 'Tm', 'Opp'],
                           columns=['player_rank'], aggfunc=np.sum)

    df_player.columns = ['_'.join(col).strip() for col in df_player.columns.values]
    cols = df_player.columns.tolist()
    df_player = df_player.reset_index(drop = False)

    grouped_df = df_player.groupby(['Tm'])
    for c in cols:
        for time in [10, 20, 30, 80]:
            try:
                df_player.loc[:, f'{c}_avg_{time}'] = grouped_df[c].transform(lambda x: x.rolling(time, min_periods=10).mean())
            except Exception as e:
                print('Error on {} : {}'.format(c, e))

    save_path = pathlib.Path.cwd().parent / 'data' / 'processed_player_df.feather'
    df_player.to_feather(str(save_path))

    return df_player


def aggregate_player_data(method = 'rolling'):

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
        for window in [5, 10, 15, 30]:
            try:
                player_df.loc[:, c + '_avg'] = grouped_df[c].transform(lambda x : x.rolling(window, min_periods=window).mean())
                # player_df.loc[:, c + 'ewm_avg'] = grouped_df[c].transform(lambda x: x.ewm(alpha = 0.9, min_periods = 10).mean())
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
    train_scaler = StandardScaler()
    valid_scaler = StandardScaler()
    df_train.loc[:, feature] = train_scaler.fit_transform(df_train[feature].values)
    df_valid.loc[:, feature] = valid_scaler.fit_transform(df_valid[feature].values)
    df_test.loc[:, feature] = valid_scaler.transform(df_test[feature].values)
    return df_train, df_valid, df_test


def generate_test_predictions(df, args):
    home = ['CHO', 'DET', 'IND', 'CLE', 'PHI', 'TOR', 'MIL', 'MIN', 'SAS', 'PHO', 'LAC', 'POR', 'SAC']
    away = ['CHI', 'HOU', 'ORL', 'BRK', 'BOS', 'MIA', 'WAS', 'ATL', 'DAL', 'DEN', 'OKC', 'MEM', 'NYK']
    for h, a in zip(home, away):
        assert h in mapping_abbrev_full
        assert a in mapping_abbrev_full
    vals = {'Home': home,
            'Away': away,
            'Opp': home,
            'Date': [args.date] * len(home),
            'unique_id': [f'{home}_{away}' for home, away in zip(home, away)],
            'Season': [2021] * len(home)}

    df_vals = pd.DataFrame(vals)
    df_vals_a = df.columns.tolist()
    cols = [c for c in df_vals_a if c not in df_vals.columns.tolist()]
    df = df.loc[:, ~df.columns.duplicated()]

    for c in cols:
        df_vals[c] = np.NaN

    tt = pd.concat([df, df_vals], axis=0, ignore_index=True)
    home_feats = df.columns[df.columns.str.contains('home')]
    away_feats = df.columns[df.columns.str.contains('away')]
    for home_feat, away_feat in zip(home_feats, away_feats):
        tt.loc[:, home_feat] = tt.groupby(['Home'])[home_feat].transform(lambda x: x.fillna(method='ffill'))
        tt.loc[:, away_feat] = tt.groupby(['Away'])[away_feat].transform(lambda x: x.fillna(method='ffill'))
    return tt

def merge_player_data(df, df_agg, features, shift_num_games):
    df = df.merge(df_agg, how='left', left_on=['Date', 'Home'], right_on=['Date', 'Tm'], suffixes=['', '_home'])
    df = df.merge(df_agg, how='left', left_on=['Date', 'Away'], right_on=['Date', 'Tm'], suffixes=['', '_away'])

    count_cols = df.columns[df.columns.str.contains('count')].tolist()
    # mappings = {c : 0 for c in count_cols}
    for c in count_cols:
        try:
            df.loc[:, c] = df.groupby(['Home'])[c].transform(lambda x: x.fillna(method='ffill'))
            df.loc[:, c] = df.groupby(['Away'])[c].transform(lambda x: x.fillna(method='ffill'))
        except Exception as e:
            print(f'Error on col: {c} with {e}')

    df_copy = df.copy()
    # TODO: Need to modify this
    df = generate_test_predictions(df_copy, arg)

    for c in count_cols:
        df.loc[:, c] = df.groupby(['Home'])[c].transform(lambda x: x.fillna(method='ffill'))
        df.loc[:, c] = df.groupby(['Away'])[c].transform(lambda x: x.fillna(method='ffill'))

    periods = [10, 20, 30]
    all_features = []
    for feat in features:
        for period, alpha in zip(periods, alphas):
            cur_feature = feat + f'_rolling_avg_{period}'
            all_features.append(cur_feature)
            df.loc[:, feat + f'_rolling_avg_{period}'] = df.groupby(['Home'])[feat].transform(
                lambda x: x.rolling(period).mean())
            if 'c_' not in feat:
                for i in range(1, shift_num_games + 1):
                    df.loc[:, feat + '_last_{}_game_played'.format(i)] = df.groupby(['unique_id'])[feat].transform(
                        lambda x: x.shift(i))
    return df, all_features, count_cols

def main(arg, seed, X_train_df, X_valid_df, X_test_df, y_train_df, y_valid_df, y_test_df, features):
    if arg.model == 'total':
        log_dir = pathlib.Path.cwd().parent / 'output' / f'tb_{arg.log_dir_folder}' / f'{arg.date}_{arg.model}_{arg.threshold}_{seed}'
    else:
        log_dir = pathlib.Path.cwd().parent / 'output' / f'tb_{arg.log_dir_folder}' / f'{arg.date}_{arg.model}_{seed}'

    args = {
        'emb_dims': cat_szs,
        'continuous_features': len(features),
        'lin_layer_sizes': arg.lin_layer_size,
        'output_size': 2,
        'embedding_dropout': 0.2,
        'linear_layer_dropout': arg.lin_layer_dropout,
        'lr': arg.lr,
        'use_embedding': True,
        'seed': seed,
        'log_dir': str(log_dir),
        'objective': arg.model,
        'swa': arg.swa,
        'loss_fn' : arg.loss_fn,
        'scheduler': arg.scheduler
    }

    if arg.model != 'regression':
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


    train_label_tensor = torch.from_numpy(y_train_df.values.astype('long')).long().view(-1)
    valid_label_tensor = torch.from_numpy(y_valid_df.values.astype('long')).long().view(-1)

    print(train_label_tensor.size())

    train_dataset = torch.utils.data.TensorDataset(cat_train_tensor,
                                                   cont_train_tensor,
                                                   train_label_tensor)

    valid_dataset = torch.utils.data.TensorDataset(cat_valid_tensor,
                                                   cont_valid_tensor,
                                                   valid_label_tensor)

    # TODO: Check if we need to add weights
    # if arg.model != 'regression':
    #     label_tr = y_train_df.to_frame('label').copy()
    #     label_tr = label_tr.assign(weights=1)
    #     label_tr.loc[label_tr['label'] == 1, 'weights'] = pos_weight
    #     label_val = y_valid_df.to_frame('label').copy()
    #     label_val = label_val.assign(weights=1)
    #     label_val.loc[label_val['label'] == 1, 'weights'] = 1
    #     all_wts = pd.concat([label_tr, label_val], axis=0, ignore_index=True)
    #     all_wts.loc[all_wts['label'] == 1, 'weights'] = pos_weight

    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    if arg.weighted_sampler:
        train_sampler = torch.utils.data.WeightedRandomSampler(label_tr['weights'].values.tolist(), label_tr.shape[0],
                                                               replacement=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, sampler=train_sampler, drop_last=True,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=512, shuffle=True,
                                                   drop_last=True, pin_memory=True)
    else:

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=512, shuffle=True,
                                                   drop_last=True, pin_memory=True)

    if arg.weighted_sampler:
        epochs = model.train(args, patience=arg.early_stopping, train_dataset=train_loader, valid_dataset=valid_loader,
                             pos_weights=pos_weight)
    else:
        epochs = model.train(args, patience=arg.early_stopping, train_dataset=train_loader, valid_dataset=valid_loader)

    dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])

    if arg.weighted_sampler:
        all_sampler = torch.utils.data.WeightedRandomSampler(all_wts['weights'].values.tolist(), all_wts.shape[0],
                                                             replacement=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, sampler=all_sampler, drop_last=True,
                                                 pin_memory=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True,
                                                 pin_memory=True)

    if (arg.model != 'regression'):
        if ((pos_count / y_train_df.shape[0]) < 0.3):
            nn_model, loss = model.train(args, patience=arg.early_stopping, num_epochs=epochs, early_stopping=False,
                                         dataset=dataloader, pos_weights=pos_weight)
        else:
            nn_model, loss = model.train(args, patience=arg.early_stopping, num_epochs=epochs, early_stopping=False,
                                         dataset=dataloader)
    else:
        nn_model, loss = model.train(args, patience=arg.early_stopping, num_epochs=epochs, early_stopping=False,
                                     dataset=dataloader)

    cont_test_tensor = torch.from_numpy(X_test_df[features].values).float()
    cat_test_tensor = torch.from_numpy(X_test_df[categorical_features].values.astype('int')).long()
    # y_test_df = pd.get_dummies(y_test_df)
    test_label_tensor = torch.from_numpy(y_test_df.values.astype('long')).long().view(-1)

    test_dataset = torch.utils.data.TensorDataset(cat_test_tensor,
                                                  cont_test_tensor,
                                                  test_label_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, drop_last=False,
                                              pin_memory=True)
    train_pred_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=False, drop_last=False,
                                                    pin_memory=True)
    valid_pred_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=512, shuffle=False, drop_last=False,
                                                    pin_memory=True)

    y_test_pred, log_preds_test = model.make_predictions(test_loader, nn_model, args)
    # y_train_pred, log_preds_train = model.make_predictions(train_pred_loader, nn_model, args)
    # y_valid_pred, log_preds_valid = model.make_predictions(valid_pred_loader, nn_model, args)

    return y_test_pred

def convert_to_dataframe(predictions, source_data):
    years = set([key[0] for key, value in predictions.items()])
    seeds = set([key[1] for key, value in predictions.items()])
    all_predictions, all_label, all_X = {}, {}, {}
    for year in years:
        predictions_year = []
        for seed in seeds:
            df = pd.DataFrame(predictions[(year, seed)], columns = [f'y_test_pred_seed_{seed}'])
            df = df.assign(year = year)
            predictions_year.append(df)
        df_label = source_data[year][-1]
        all_label[year] = df_label
        X_test_df = source_data[year][2]
        all_X[year] = X_test_df
        df_year = pd.concat(predictions_year, axis = 1, ignore_index = False)
        all_predictions[year] = df_year
    df_all = pd.concat(list(all_predictions.values()), axis = 0, ignore_index = True)
    X_test_df = pd.concat(list(all_X.values()), axis = 0, ignore_index = True)
    all_label_df = pd.concat(list(all_label.values()), axis = 0, ignore_index = True)
    return df_all, X_test_df, all_label_df

def arg_parse():
    parser = argparse.ArgumentParser(description = 'Player Model')
    parser.add_argument('-production', default = '1', type = int, help = 'production or train')
    parser.add_argument('-date', type = str, default='2021-01-22', help = 'prediction date')
    parser.add_argument('-model', type = str, default = 'moneyline', help = 'options: moneyline vs total')
    parser.add_argument('-threshold', type = int, default = '50', help = 'only used if model = total')
    parser.add_argument('-load_processed_data', type = int, default = 1, help = 'load preprocessed data')
    parser.add_argument('-save_processed_data', type = int, default = 0, help = 'save preprocessed data')
    parser.add_argument('-lin_layer_size', nargs='+', type=int, default = [512, 256], help = 'linear layers')
    parser.add_argument('-lin_layer_dropout', nargs = '+', type = float, default = [0.3, 0.3], help = 'dropout for each lin layer')
    parser.add_argument('-log_dir_folder', default = 'reg_10_20_player_model_uniqid_cos_sch_512_256_d_0.3_b_512_l1', type = str, help = 'log_dir_folder')
    parser.add_argument('-early_stopping', default = 50, type = int, help = 'early_stopping')
    parser.add_argument('-valid_criteria', default = 'accuracy', type = str, help = 'accuracy vs loss')
    parser.add_argument('-swa', default = 0, type = int, help = 'swa')
    parser.add_argument('-lr', default = 5e-4, type = float, help = 'lr')
    parser.add_argument('-weighted_sampler', type = int, default = 0, help = 'weighted sampler')
    parser.add_argument('-scheduler', default = 'cosine', type = str, help = 'cosine_annealing vs stepLR')
    parser.add_argument('-save_output', default = '0', type = int, help = 'save predictions')
    parser.add_argument('-load_predictions', default = '0', type = int, help = 'load predictions')
    parser.add_argument('-use_individual_player', default = '1', type = int)
    parser.add_argument('-loss_fn', default = 'gambler_loss', type = str, help = 'cross_entropy vs gambler_loss')
    parser.add_argument('-feats', default = 'rolling', type = str, help = 'rolling vs ewm')
    return parser


if __name__ == '__main__':
    arg = arg_parse()
    arg = arg.parse_args()

    print(f'Args are: {arg}')
    print(f'Prediction date is {arg.date}')
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
        df = df.sort_values(by=['Date']).reset_index(drop=True)
        df = df.drop_duplicates(['unique_id', 'Date'])
        df = df.fillna(0.0)

        home_feats = df.columns[df.columns.str.contains('home')]
        away_feats = df.columns[df.columns.str.contains('away')]
        for home_feat, away_feat in zip(home_feats, away_feats):
            # TODO : groupby home and away
            df.loc[:, home_feat] = df.groupby(['Home'])[home_feat].transform(lambda x: x.shift(1))
            df.loc[:, away_feat] = df.groupby(['Away'])[away_feat].transform(lambda x: x.shift(1))
            # TODO : groupby unique_id
            # df.loc[:, home_feat] = df.groupby(['unique_id'])[home_feat].transform(lambda x: x.shift(1))
            # df.loc[:, away_feat] = df.groupby(['unique_id'])[away_feat].transform(lambda x: x.shift(1))

        # TODO: Uncomment this
        game_path = pathlib.Path.cwd().parent / 'data' / 'game_play_data.feather'

        df['Date'] = pd.to_datetime(df['Date'])
        shift_num_games = 5
        features = df.columns[df.columns.str.contains('home') | df.columns.str.contains('away')].tolist()
        # generate and process player data
        if arg.use_individual_player:
            df_agg = aggregate_player_data_individual()
            # col_renamed = df_agg.columns[:2].tolist() + ['count_' + c for c in df_agg.columns[2:].tolist()]
            # df_agg.columns = col_renamed
        else:
            df_agg = aggregate_player_data()
            col_renamed = df_agg.columns[:2].tolist() + ['count_' + c for c in df_agg.columns[2:].tolist()]
            df_agg.columns = col_renamed

        df, all_features, count_cols = merge_player_data(df, df_agg, features, shift_num_games)

        if arg.save_processed_data:
            data_dir = pathlib.Path.cwd().parent / 'data'
            filename = str(data_dir) + f'/player_model_features_uniqid.feather'
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df.to_feather(filename)
            all_feature_fname = str(data_dir) + '/all_features_uniqid.pickle'
            save_obj(all_feature_fname, all_features)
            if not count_cols:
                count_cols_fname = str(data_dir) + '/count_cols_uniqid.pickle'
                save_obj(count_cols_fname, count_cols)
            features_fname = str(data_dir) + '/features_uniqid.pickle'
            save_obj(features_fname, features)
    else:
        data_dir = pathlib.Path.cwd().parent / 'data'
        filename = str(data_dir) + '/player_model_features_uniqid.feather'
        all_feature_fname = str(data_dir) + '/all_features_uniqid.pickle'
        count_cols_fname = str(data_dir) + '/count_cols_uniqid.pickle'
        features_fname = str(data_dir) + '/features_uniqid.pickle'
        df = pd.read_feather(filename)
        all_features = load_obj(all_feature_fname)
        all_features = [feat for feat in all_features if feat in df.columns]
        features = load_obj(features_fname)
        count_cols = load_obj(count_cols_fname)
        df = utils.generate_labels(df, arg)
        df = df.assign(year=lambda x: x['Date'].dt.year)
        all_features += ['year']

    cols = [i for i in df.columns if i in all_features]
    # rolling_feats = [feat for feat in df.columns if 'avg' in feat]
    features = features + all_features + df.columns[df.columns.str.contains('_game_played')].tolist() + count_cols
    feature_set = set(features)
    rolling_feats = [feat for feat in df.columns if ('player' in feat) and ('avg' not in feat)]
    features += rolling_feats
    features = list(set(features))
    # features = [feat for feat in features if 'rolling' not in feat]

    bias = 'score_diff' in features
    print(f'is score_diff in Features ? {bias}')

    df = df.reset_index(drop=True)
    df = df.fillna(0.0)
    print('shape of data before dropping is {}'.format(df.shape))
    df = df.dropna(subset=features, axis=0)
    print('shape of data after dropping is {}'.format(df.shape))
    print(f'Length of features is {len(features)}')

    df, categorical_features = encode_categorical_data(df)

    cat_szs = [(df[c].nunique() + 1, min(50, (df[c].nunique()) // 2)) for c in categorical_features]

    predict = df.loc[df['Date'] == arg.date, :].reset_index(drop = True)
    df = df.loc[df['Date'] != arg.date, :].reset_index(drop=True)

    prediction_date = arg.date
    predict_bool = arg.production
    #
    years = [i for i in range(2015, 2021)]
    predictions = {}
    source_data = {}
    if arg.load_predictions == False:
        for seed in range(1, 5):
            if predict_bool == False:
                for year in years:
                    X_train_df = df.loc[(df['Season'] < (year - 2)) & (df['Season'] > first_year), :].reset_index(drop=True)
                    y_train_df = df.loc[(df['Season'] < (year - 2)) & (df['Season'] > first_year), 'target'].reset_index(drop=True)

                    X_valid_df = df.loc[(df['Season'] >= (year - 2)) & (df['Season'] < year), :].reset_index(drop=True)
                    y_valid_df = df.loc[(df['Season'] >= (year - 2)) & (df['Season'] < year), 'target'].reset_index(drop=True)

                    X_test_df = df.loc[(df['Season'] == year) & (df['Date'] != prediction_date), :].reset_index(drop=True)
                    y_test_df = df.loc[(df['Season'] == year) & (df['Date'] != prediction_date), 'target'].reset_index(drop=True)
                    y_test_pred = main(arg, seed, X_train_df, X_valid_df, X_test_df, y_train_df, y_valid_df, y_test_df, features)
                    predictions[(year, seed)] = y_test_pred
                    if year not in source_data:
                        source_data[year] = (X_train_df, X_valid_df, X_test_df, y_train_df, y_valid_df, y_test_df)
            else:
                X_train_df = df.loc[(df['Season'] < last_year) & (df['Season'] > first_year), :].reset_index(drop=True)
                y_train_df = df.loc[(df['Season'] < last_year) & (df['Season'] > first_year), 'target'].reset_index(drop=True)

                X_valid_df = df.loc[(df['Season'] >= last_year), :].reset_index(drop=True)
                y_valid_df = df.loc[(df['Season'] >= last_year), 'target'].reset_index(drop=True)

                X_test_df = predict.reset_index(drop=True)
                y_test_df = predict.loc[:, 'target'].reset_index(drop=True)

                y_test_pred = main(arg, seed, X_train_df, X_valid_df, X_test_df, y_train_df, y_valid_df, y_test_df, features)
                predictions[(last_year, seed)] = y_test_pred

        if not arg.production:
            df_preds, X_test_df, all_label_df = convert_to_dataframe(predictions, source_data)
            df_preds = df_preds.loc[:,~df_preds.columns.duplicated()]
            pred_cols = df_preds.columns[df_preds.columns.str.contains('y_test_pred_seed')].tolist()
            cols_to_keep = pred_cols + ['ytest_pred_mean', 'target', 'year', 'Date', 'Home', 'Away', 'Tm_score', 'Opp_score']
            df_preds['ytest_pred_mean'] = df_preds[pred_cols].mean(axis = 1)
            y_df = pd.concat([df_preds, all_label_df], axis=1, ignore_index=False)
            df_all = pd.concat([y_df, X_test_df], axis=1, ignore_index=False)
            df_filt = df_all[cols_to_keep]
            df_filt_orig = df[[col for col in cols_to_keep if col in df]]
            if arg.save_output:
                # source_output = pathlib.Path.cwd().parent / 'output' / f'{arg.log_dir_folder}_source_data.pkl'
                # pred_output = pathlib.Path.cwd().parent / 'output' / f'{arg.log_dir_folder}_pred_output.pkl'
                # save_obj(str(source_output), source_data)
                # save_obj(str(pred_output), predictions)
                source_name = pathlib.Path.cwd().parent / 'output' / f'{arg.log_dir_folder}_source_data.feather'
                pred_name = pathlib.Path.cwd().parent / 'output' / f'{arg.log_dir_folder}_pred_output.feather'
                loc_idx = (X_test_df['Location'] != 0)
                y_df, X_test_df = y_df.loc[loc_idx], X_test_df.loc[loc_idx]
                tm_idx = (X_test_df['Tm'] != 0)
                y_df, X_test_df = y_df.loc[tm_idx], X_test_df.loc[tm_idx]
                y_df, X_test_df = y_df.reset_index(drop = True), X_test_df.reset_index(drop = True)
                X_test_df.to_feather(str(source_name))
                df_preds.to_feather(str(pred_name))
    else:
        source_output = pathlib.Path.cwd().parent / 'output' / f'{arg.log_dir_folder}_source_data.pkl'
        pred_output = pathlib.Path.cwd().parent / 'output' / f'{arg.log_dir_folder}_pred_output.pkl'
        source_data = load_obj(str(source_output))
        predictions = load_obj(str(pred_output))

        df_preds, X_test_df, all_label_df = convert_to_dataframe(predictions, source_data)
        cols_to_keep = ['y_test_pred_seed_1','target','year',
                        'Date', 'Home', 'Away', 'Tm_score', 'Opp_score']
        y_df = pd.concat([df_preds, all_label_df], axis=1, ignore_index=False)
        df_all = pd.concat([y_df, X_test_df], axis=1, ignore_index=False)
        df_filt = df_all[cols_to_keep]
        df_filt_orig = df[cols_to_keep[1:]]

    predictions_fname = pathlib.Path.cwd().parent / 'output' / f'{arg.date}_predictions' / f'{arg.loss_fn}' / f'predictions_{arg.loss_fn}.csv'
    predictions_fname_dir = pathlib.Path.cwd().parent / 'output' / f'{arg.date}_predictions' / f'{arg.loss_fn}'
    if not predictions_fname_dir.is_dir():
        predictions_fname_dir.mkdir(parents = True)
    df_preds = {}
    df_preds[0] = np.concatenate([predictions[(2020, i)][:, 0].reshape(-1, 1) for i in range(1, 5)], axis = 1)
    df_preds[1] = np.concatenate([predictions[(2020, i)][:, 1].reshape(-1, 1) for i in range(1, 5)], axis = 1)
    df_avg = {}
    df_avg['0_avg'] = np.mean(df_preds[0], axis = 1)
    df_avg['1_avg'] = np.mean(df_preds[1], axis = 1)
    df_avg_combined = pd.DataFrame.from_dict(df_avg)
    df_avg_combined['unique_id'] = X_test_df['unique_id']
    for j in range(1, 5):
        df_avg_combined[f'nn{j}_pred=0'] = df_preds[0][:, j - 1]
        df_avg_combined[f'nn{j}_pred=1'] = df_preds[1][:, j - 1]
    df_avg_combined.to_csv(predictions_fname)


    # ppp = {k: v.ravel() for k, v in predictions.items()}
    # pr = pd.DataFrame(ppp)
    # pr.columns = ['nn0', 'nn1', 'nn2', 'nn3']
    # probs = pr.mean(axis = 1)
    # probs_df_output = pd.concat([X_test_df, pr], axis = 1)
    # path = pathlib.Path.cwd().parent / 'output' / f'{arg.date}_predictions'
    # if path.exists() == False:
    #     path.mkdir()
    # probs_df_output.to_csv(str(path) + f'/{arg.date}_predictions.csv')

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
