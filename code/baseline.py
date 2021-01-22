import sys
sys.path.append('./')
import pandas as pd
import numpy as np
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
from config import mapping
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def process_data(df, team):
    keep_cols = ['Date', 'Location', 'Opp', 'Home', 'Away',
                 'Tm_score', 'Opp_score']
    keep_cols.append('unique_id') # unique identifier by '{Home_team}_{Away_team}'

    df = df.rename(columns={df.columns[0] : 'Date'})
    ## 1 is Current Team, #2 is Opponent Team
    ## Feature processing
    df.loc[df['Location'] == '@', 'Location'] = 'Away'
    df.loc[~(df['Location'] == 'Away'), 'Location'] = 'Home'
    df = df.drop(columns=['3P%1', '3P%2'], axis = 1)
    df['DRB1'] = df['TRB1'] - df['ORB1']
    df['DRB2'] = df['TRB2'] - df['ORB2']
    tmp = df.select_dtypes(include=['float', 'int']).cumsum(axis=0 )
    cum_cols = ['c_'+ str(col) for col in tmp.columns]
    tmp.columns = cum_cols
    df2 = pd.concat([df, tmp], axis=1)

    df2.loc[df2['Location'] == 'Home','Home'] = team
    df2.loc[df2['Location'] == 'Home','Away'] = df2.loc[df2['Location'] == 'Home','Opp']
    df2.loc[df2['Location'] == 'Away','Home'] = df2.loc[df2['Location'] == 'Away','Opp']
    df2.loc[df2['Location'] == 'Away','Away'] = team
    df2 = df2.assign(unique_id = lambda x : x['Home'] + '_' + x['Away'], target = np.NaN, score_diff = np.NaN)

    df2.loc[df2['Location'] == 'Home', 'target'] = ((df['Tm_score'] - df['Opp_score']) > 0)
    df2.loc[df2['Location'] == 'Home', 'score_diff'] = (df['Tm_score'] - df['Opp_score'])
    df2.loc[df2['Location'] == 'Away', 'target'] = ((df['Opp_score'] - df['Tm_score']) > 0)
    df2.loc[df2['Location'] == 'Away', 'score_diff'] = (df['Opp_score'] - df['Tm_score'])

    return df2

def process_data_all(path, teams, first_year, last_year):
    results = []
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        filename = filename.split('/')[-1]
        filename = filename.replace('.csv', '')
        try:
            team = filename.split('_')[0]
            year = int(filename.split('_')[1])
        except:
            print('Error on {}'.format(filename))
        try:
            df = pd.read_csv(path + team + '_' + str(year) + '.csv')
        except:
            continue
        df_mod = process_data(df, team)
        df_mod['Season'] = year  # Season is the ending year of season (e.g. 2015 is 2014-2015 season)
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

if __name__ == '__main__':

    dir = '../data/Basketball/Team/gamelog/'
    odds_data_path = '../data/scraped_odds_data.csv'

    teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET',
             'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
             'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS',
             'TOR', 'UTA', 'WAS']

    first_year = 1985
    last_year = 2019

    df = process_data_all(dir, teams, first_year, last_year)

    # verify data format by taking few examples
    test_date = '2017-10-18'
    df_2018 = df.loc[(df['Date'] == test_date), :]
    df_2018 = df_2018.drop_duplicates(['unique_id'])

    # prepare oracle features
    features = ['FG', 'FGA', '3P', 'FT', 'TOV']
    oracle_features = [i + str(j) for i in features for j in range(1,3)]

    df = df.sort_values(by = ['Date']).reset_index(drop = True)
    df = df.drop_duplicates(['unique_id', 'Date'])
    df = df.fillna(0.0)
    # predict from 2014 - 2019; train test split (train < 2014)
    X_train = df.loc[(df['Season'] < 2014), oracle_features]
    y_train = df.loc[(df['Season'] < 2014), ['target']]

    X_test = df.loc[(df['Season'] >= 2014), oracle_features]
    y_test = df.loc[(df['Season'] >= 2014), ['target']]

    X = pd.concat([X_train, X_test], axis = 0, ignore_index= True)
    y = pd.concat([y_train, y_test], axis = 0, ignore_index= True)

    knn = KNeighborsClassifier(n_neighbors=3, n_jobs = -1).fit(X, y)
    y_pred = knn.predict(X)
    print('Oracle Accuracy = {}'.format(accuracy_score(y, y_pred)))

    np.random.seed(0)
    # naive prediction using random guessing
    # y_train_naive = np.random.choice([0,1], size = y_train.shape[0])
    # y_test_naive = np.random.choice([0,1], size = y_test.shape[0])
    df['naive_pred'] = (df['Location'] == 'Home') # always guess that the Home team will win
    print('Naive Accuracy = {}'.format(accuracy_score(df['target'], df['naive_pred'])))




    # print('Test Set Naive Accuracy = {}'.format(accuracy_score(y_test, y_test_naive)))

    # odds_df = pd.read_csv(odds_data_path, header = 0)
    # odds_df = process_odds_data(odds_df, df)
    #
    # df['win_loss_id'] = df['unique_id']
    # df.loc[~(df['target']), 'win_loss_id'] = (df.loc[~(df['target']), 'unique_id']
    #                                             .apply(lambda x : '_'.join(reversed(x.split('_')))))
    # df['Date'] = pd.to_datetime(df['Date'])
    #
    # df['month'] = df['Date'].dt.month
    # df_merge = df.merge(odds_df, how = 'left', left_on = ['Season', 'win_loss_id'],
    #                     right_on = ['season', 'win_loss_id'])
    #
    # # predict from 2014 - 2019; train test split
    # X_train = df_merge.loc[(df_merge['Season'] < 2014), oracle_features]
    # y_train = df_merge.loc[(df_merge['Season'] < 2014), 'target']
    # df_train = df_merge.loc[(df_merge['Season'] < 2014), :].reset_index(drop = True)
    #
    # X_test = df_merge.loc[(df_merge['Season'] >= 2014), oracle_features]
    # y_test = df_merge.loc[(df_merge['Season'] >= 2014), 'target']
    # df_test = df_merge.loc[(df_merge['Season'] >= 2014), :].reset_index(drop = True)
    #
    # lr = LogisticRegression().fit(X_train, y_train)
    # y_train_pred = lr.predict(X_train)
    # y_test_pred = lr.predict(X_test)
    # print('Train Set Oracle Accuracy = {}'.format(accuracy_score(y_train, y_train_pred)))
    # print('Test Set Oracle Accuracy = {}'.format(accuracy_score(y_test, y_test_pred)))
    #
    # df_train['y_pred_oracle'] = y_train_pred
    # df_test['y_pred_oracle'] = y_test_pred

