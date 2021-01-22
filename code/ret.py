import feather
import pandas as pd
import numpy as np
import baseline
import matplotlib.pyplot as plt


if __name__ == '__main__':

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

    X_test = feather.read_dataframe('../data/X_test_df.feather')
    df = X_test
    y_pred = feather.read_dataframe('../data/nn_test_predictions.feather')
    y_pred_prob = feather.read_dataframe('../data/nn_test_probability_predictions.feather')
    y_test = feather.read_dataframe('../data/y_test_df.feather')
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
        # if (df_rets_drop.loc[i]['predictions'] == True) and (df_rets_drop.loc[i]['odds1'] > -100):
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
        # if (df_rets_drop.loc[i]['predictions'] == False) and (df_rets_drop.loc[i]['odds2'] > -100):
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

    print('Total Profit Is {}'.format(profit.sum()))

    result = pd.DataFrame({'profit': cum_profit}, index=date)

    result.plot(y = 'profit', rot = 45, title = 'Cumulative Profit From $100')
    plt.savefig('../output/prob_0.3_0.7.jpg', format = 'jpg')
    plt.show()