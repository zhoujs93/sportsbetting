import sys
sys.path.append('./')
import glob
import re, typing
from config import mapping
import pandas as pd
import numpy as np
import baseline
import matplotlib.pyplot as plt

def process_data(df, team, args):
    keep_cols = ['Date', 'Location', 'Opp', 'Home', 'Away',
                 'Tm_score', 'Opp_score']
    keep_cols.append('unique_id') # unique identifier by '{Home_team}_{Away_team}'

    df = df.rename(columns={df.columns[0] : 'Date'})
    ## 1 is Current Team, #2 is Opponent Team
    ## Feature processing
    df.loc[df['Location'] == '@', 'Location'] = 'Away'
    df.loc[~(df['Location'] == 'Away'), 'Location'] = 'Home'
    # need to map feature 1 to home team, and feature 2 to away team

    df['DRB1'] = df['TRB1'] - df['ORB1']
    df['DRB2'] = df['TRB2'] - df['ORB2']
    tmp = df.select_dtypes(include=['float', 'int']).columns.tolist()
    df_grouped = df.groupby(['Season'])
    for col in tmp:
        df.loc[:, f'c_{col}'] = df_grouped[col].transform(lambda x: x.cumsum(axis = 0))
    df2 = make_team_mapping(df)

    df2.loc[df2['Location'] == 'Home','Home'] = team
    df2.loc[df2['Location'] == 'Home','Away'] = df2.loc[df2['Location'] == 'Home','Opp']
    df2.loc[df2['Location'] == 'Away','Home'] = df2.loc[df2['Location'] == 'Away','Opp']
    df2.loc[df2['Location'] == 'Away','Away'] = team
    df2 = df2.assign(unique_id = lambda x : x['Home'] + '_' + x['Away'], target = np.NaN, score_diff = np.NaN)

    if args.model == 'moneyline':
        df2.loc[df2['Location'] == 'Home', 'target'] = ((df['Tm_score'] - df['Opp_score']) > 0)
        df2.loc[df2['Location'] == 'Home', 'score_diff'] = (df['Tm_score'] - df['Opp_score'])
        df2.loc[df2['Location'] == 'Away', 'target'] = ((df['Opp_score'] - df['Tm_score']) > 0)
        df2.loc[df2['Location'] == 'Away', 'score_diff'] = (df['Opp_score'] - df['Tm_score'])
    elif args.model == 'regression':
        df2 = df2.assign(target = lambda x: x['Tm_score'] + x['Opp_score'])
    else:
        df2 = df2.assign(score_total = lambda x: x['Tm_score'] + x['Opp_score'])
        df2.loc[(df2['score_total'] > args.threshold), 'target'] = 1
        df2.loc[(df2['score_total'] <= args.threshold), 'target'] = 0
        df2 = df2.drop(['score_total'], axis = 1)
    return df2

def generate_labels(df2, args):
    if args.model == 'moneyline':
        df2.loc[df2['Location'] == 'Home', 'target'] = ((df2['Tm_score'] - df2['Opp_score']) > 0)
        df2.loc[df2['Location'] == 'Home', 'score_diff'] = (df2['Tm_score'] - df2['Opp_score'])
        df2.loc[df2['Location'] == 'Away', 'target'] = ((df2['Opp_score'] - df2['Tm_score']) > 0)
        df2.loc[df2['Location'] == 'Away', 'score_diff'] = (df2['Opp_score'] - df2['Tm_score'])
    elif args.model == 'regression':
        df2 = df2.assign(target = lambda x: x['Tm_score'] + x['Opp_score'])
    elif args.model == 'binary_regression':
        df2 = df2.assign(target_total=lambda x: x['Tm_score'] + x['Opp_score'], target = 0)
        df2.loc[df2['ytest_pred_mean'] > df2['target_total'], 'target'] = 1
        df2.loc[df2['ytest_pred_mean'] <= df2['target_total'], 'target'] = 0
        df2 = df2.drop(['target_total'], axis = 1)
    else:
        df2 = df2.assign(score_total = lambda x: x['Tm_score'] + x['Opp_score'])
        df2.loc[(df2['score_total'] > args.threshold), 'target'] = 1
        df2.loc[(df2['score_total'] <= args.threshold), 'target'] = 0
        df2 = df2.drop(['score_total'], axis = 1)
    return df2

def make_team_mapping(df):
    team_one_cols = df.columns[df.columns.str.contains('1')]
    team_two_cols = df.columns[df.columns.str.contains('2')]
    assert len(team_one_cols) == len(team_two_cols)
    for col, col2 in zip(team_one_cols, team_two_cols):
        new_col = col.replace('1', '_home')
        new_col_away = col.replace('1', '_away')
        df.loc[:, new_col] = np.NaN
        df.loc[(df['Location'] == 'Home'), new_col] = df.loc[(df['Location'] == 'Home'), col]
        df.loc[(df['Location'] == 'Away'), new_col] = df.loc[(df['Location'] == 'Away'), col2]
        df.loc[(df['Location'] == 'Away'), new_col_away] = df.loc[(df['Location'] == 'Away'), col]
        df.loc[(df['Location'] == 'Home'), new_col_away] = df.loc[(df['Location'] == 'Home'), col2]

    df = df.drop(team_two_cols.tolist() + team_one_cols.tolist(), axis = 1)
    return df

def process_data_all(path, args):
    results = []
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        filename = filename.split('/')[-1]
        filename = filename.replace('.csv', '')
        try:
            team = filename.split('_')[0]
            year = int(filename.split('_')[1])
        except Exception as e:
            print('Error on {} : {}'.format(filename, e))
        try:
            df = pd.read_csv(path + team + '_' + str(year) + '.csv')
        except Exception as e:
            print(f'Error for {filename} : {e}')
            continue
        df['Season'] = year  # Season is the ending year of season (e.g. 2015 is 2014-2015 season)
        df_mod = process_data(df, team, args)
        results.append(df_mod)

    frame = pd.concat(results, axis=0, ignore_index=True)
    return frame

def process_odds_data(odds_df, df):
    def f(x):
        if 'Today' in x:
            x = x.replace('Today, ', '')
            x += '2019'
        if 'Yesterday' in x:
            x = x.replace('Yesterday, ', '')
            x += '2019'
        x = re.sub(' +', ' ', x)
        return x

    odds_df['date'] = odds_df['date'].apply(f)

    odds_df['date'] = pd.to_datetime(odds_df['date'])
    odds_df = odds_df.sort_values(['date'])

    odds_df['team1'] = odds_df['team1'].map(mapping)
    odds_df['team2'] = odds_df['team2'].map(mapping)
    odds_df = odds_df.replace({'odds1' : '-', 'odds2' : '-'}, '0')
    odds_df['odds1'] = odds_df['odds1'].astype('float')
    odds_df['odds2'] = odds_df['odds2'].astype('float')
    odds_df['odds'] = (odds_df[['odds1','odds2']].max(axis = 1)) / 100
    odds_df['min_odds'] = (odds_df[['odds1','odds2']].min(axis = 1)) / 100

    odds_df = odds_df.assign(win_loss_id = lambda x : x['team1'] + '_' + x['team2'])
    odds_df['season'] = odds_df['date'].dt.year
    odds_df['month'] = odds_df['date'].dt.month
    odds_df.loc[odds_df['month'] > 8, 'season'] = odds_df['season'] + 1

    df_max_date = df['Date'].max()
    odds_df = odds_df.loc[(odds_df['date'] <= df_max_date), :].reset_index(drop = True)
    return odds_df


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


def backtest(y_pred, y_pred_prob, model_name):

    process = False
    dir = '../data/Basketball/Team/gamelog/'
    odds_data_path = '../data/scraped_odds_data.csv'

    teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET',
             'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
             'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS',
             'TOR', 'UTA', 'WAS']

    mapping = {
        'Philadelphia 76ers': 'PHI',
        'Denver Nuggets': 'DEN',
        'Golden State Warriors': 'GSW',
        'Milwaukee Bucks': 'MIL',
        'Toronto Raptors': 'TOR',
        'Los Angeles Clippers': 'LAC',
        'San Antonio Spurs': 'SAS',
        'Houston Rockets': 'HOU',
        'Portland Trail Blazers': 'POR',
        'Utah Jazz': 'UTA',
        'Detroit Pistons': 'DET',
        'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL',
        'Indiana Pacers': 'IND',
        'Brooklyn Nets': 'BRK',
        'Boston Celtics': 'BOS',
        'Charlotte Hornets': 'CHO',
        'Los Angeles Lakers': 'LAL',
        'Sacramento Kings': 'SAC',
        'Phoenix Suns': 'PHO',
        'Dallas Mavericks': 'DAL',
        'New Orleans Pelicans': 'NOP',
        'Atlanta Hawks': 'ATL',
        'Miami Heat': 'MIA',
        'Washington Wizards': 'WAS',
        'Minnesota Timberwolves': 'MIN',
        'New York Knicks': 'NYK',
        'Chicago Bulls': 'CHI',
        'Memphis Grizzlies': 'MEM',
        'Cleveland Cavaliers': 'CLE',
    }

    X_test = pd.read_feather('../data/X_test_df.feather')
    df = X_test
    y_test = pd.read_feather('../data/y_test_df.feather')
    X_test['predictions'] = y_pred
    X_test['pred_prob'] = y_pred_prob
    dodd = pd.read_csv('../data/scraped_odds_data.csv', header=0)
    monthmap = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}

    odds_data = baseline.process_odds_data(dodd, X_test[['Date', 'Location','Home','Away', 'target', 'predictions',
                                                         'unique_id','W/L', 'Tm_score', 'Opp_score']])

    test_df = X_test[['Date', 'Location','Home','Away', 'target', 'pred_prob',
                      'predictions', 'unique_id','W/L', 'Tm_score', 'Opp_score']].copy()

    odds_data = odds_data.assign(unique_id = lambda x : x['team1'] + '_' + x['team2'])

    test_df = test_df.assign(home_score = lambda x : x['Tm_score'], away_score = lambda x : x['Opp_score'])
    test_df.loc[(test_df['Location'] == 'Away'), 'home_score'] = test_df.loc[(test_df['Location'] == 'Away'), 'Opp_score']
    test_df.loc[(test_df['Location'] == 'Away'), 'away_score'] = test_df.loc[(test_df['Location'] == 'Away'), 'Tm_score']

    test_df['home_score'] = test_df['home_score'].astype(int).astype(str)
    test_df['away_score'] = test_df['away_score'].astype(int).astype(str)
    test_df = test_df.assign(score = lambda x : x['home_score'] + ':' + x['away_score'])

    odds_data['new_score'] = odds_data['score'].apply(lambda x : x.replace('OT', ''))

    df_rets = test_df.merge(odds_data, how = 'left', left_on = ['unique_id', 'score'], right_on = ['unique_id', 'new_score'])

    df_rets_drop = df_rets.dropna(subset=['odds1', 'odds2'])

    # df_rets_drop = df_rets_drop.loc[(df_rets_drop['odds1'] > -500)
    #                                 & (df_rets_drop['odds2'] > -500)
    #                                 & (df_rets_drop['odds1'] != 0)
    #                                 & (df_rets_drop['odds2'] != 0), :].reset_index(drop = True)

    # df_rets_drop = df_rets_drop.loc[(df_rets_drop['pred_prob'] > 0.8) | (df_rets_drop['pred_prob'] < 0.2), :].reset_index(drop = True)

    mark = 0
    profit = []
    date = []
    for i in df_rets_drop.index:
        if (df_rets_drop.loc[i]['predictions'] == True) and (df_rets_drop.loc[i]['pred_prob'] > 0.7) and (df_rets_drop.loc[i]['odds1'] > -100):
        # if (df_rets_drop.loc[i]['predictions'] == True):
            mark += 100
            if df_rets_drop.loc[i]['target'] == True:
                if df_rets_drop.loc[i]['odds1'] < 0:
                    earned = -100 * 100 / df_rets_drop.loc[i]['odds1']
                    # profit += -100 * 100 / df_rets_drop.loc[i]['odds1']
                else:
                    earned =  df_rets_drop.loc[i]['odds1']
            else:
                earned = -100
                # profit -= 100
            date.append(df_rets_drop.loc[i]['Date'])
            profit.append(earned)
        if (df_rets_drop.loc[i]['predictions'] == False) and (df_rets_drop.loc[i]['pred_prob'] < 0.3) and (df_rets_drop.loc[i]['odds2'] > -100):
        # if (df_rets_drop.loc[i]['predictions'] == False):
            mark += 100
            if df_rets_drop.loc[i]['target'] == True:
                earned = -100
                # profit -= 100
            else:
                if df_rets.loc[i]['odds2'] < 0:
                    earned = -100* 100 / df_rets_drop.loc[i]['odds2']
                    # profit += -100* 100 / df_rets_drop.loc[i]['odds2']
                else:
                    earned = df_rets_drop.loc[i]['odds2']
            date.append(df_rets_drop.loc[i]['Date'])
            profit.append(earned)

    profit = np.array(profit)
    cum_profit = np.array(profit).cumsum()

    print('Model = {} : Total Profit Is {}'.format(model_name, profit.sum()))

    result = pd.DataFrame({'profit': cum_profit}, index=date)

    result.plot(y = 'profit', title = 'Cumulative Profit From $100', figsize = (8,8))
    plt.savefig('../output/{}_prob_0.3_0.7.jpg'.format(model_name), format = 'jpg')
    plt.show()

    return profit, result