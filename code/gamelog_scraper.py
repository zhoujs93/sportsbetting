# %%
import urllib.request
import requests
from bs4 import BeautifulSoup, Comment
import numpy as np
import pandas as pd
import pathlib
from functools import reduce
import argparse
import time

# %%
# teams = ['Atlanta Hawks','Boston Celtics','Brooklyn Nets','Chicago Bulls',
#          'Charlotte Hornets','Cleveland Cavaliers', 'Dallas Mavericks',
#          'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors',
#          'Houston Rockets', 'Indiana Pacers', 'Los Angeles Clippers',
#          'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
#          'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans',
#          'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic',
#          'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers',
#          'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors',
#          'Utah Jazz', 'Washington Wizards']

teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DAL', 'DEN', 'DET',
         'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
         'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS',
         'TOR', 'UTA', 'WAS']


# Charlotte Hornets only goes back to 1989 (1988-1989 season)

# Grizzlies: only goes back to 1995, from 1995-2001 is Vancouver Grizzlies

# Miami goes back to 1989 as well


# Teams added in 1989: Hornets, Miami heat, minnesota timberwolves, orlando magic
# Teams added in 1995: Toronto Raptors, vancouver grizzlies

# Pelicans: 2013 to present, previously New Orleans hornets, founded in 2002

# Thunder, before 2008 was Seattle Supersonics

# url = 'https://www.basketball-reference.com/teams/WAS/2018/gamelog/'
url1 = 'https://www.basketball-reference.com/teams/'
url2 = '/gamelog/'
first_year = 1985
last_year = 2022


# %%
# team = 'ATL'
# year = 1987
# team_aka = team
# url = url1 + team_aka + '/' + str(year) + url2
# temp = Scrap(url)
# temp.to_csv(team + '_' + str(year) + '.csv')
# print(str(year))
# %%

# %%

def Scrap(url):
    tags = ['Location', 'Opp', 'W/L', 'Tm_score', 'Opp_score',
            'FG1', 'FGA1', 'FG%1', '3P1', '3PA1', '3P%1', 'FT1', 'FTA1', 'FT%1',
            'ORB1', 'TRB1', 'AST1', 'STL1', 'BLK1', 'TOV1', 'PF1', 'FILLER',
            'FG2', 'FGA2', 'FG%2', '3P2', '3PA2', '3P%2', 'FT2', 'FTA2', 'FT%2',
            'ORB2', 'TRB2', 'AST2', 'STL2', 'BLK2', 'TOV2', 'PF2']
    n = len(tags)

    r = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(r, 'html.parser')
    table = soup.find("table")

    table_body = table.find('tbody')
    rows = table_body.find_all('tr')

    frame = []
    datetemp = []
    tmp = 0
    dfs = []
    for row in rows:

        temp = []
        line = row.get_text().strip().split()
        tag = row.find_all("td")
        line = [v.get_text().strip().replace(",", "") for v in tag]

        try:
            datetemp.append(line[1])

            for i in range(2, 5):
                temp.append(line[i])
        except:
            continue
        for i in range(5, len(line)):
            try:
                temp.append(float(line[i]))
            except:
                temp.append(np.NaN)

        temp = np.array(temp)
        temp = np.resize(temp, (1, n))

        if temp[0][0] != 0:
            frame.append(temp[0])
        tmp += 1
    frame = pd.DataFrame(frame, columns=tags, index=datetemp)
    frame.drop(columns='FILLER', inplace=True)
    dfs.append(frame)

    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    comments = soup.find_all(string = lambda text: isinstance(text, Comment))
    tables = [comment for comment in comments if 'table_outer_container' in str(comment)]
    if len(tables) == 0:
        return frame
    else:
        comment = tables[0]
        table = BeautifulSoup(str(comment), 'html.parser').find('table').find('tbody')
        playoff_rows = table.find_all('tr')

        playoff_frame = []
        playoff_datetemp = []
        playoff_tmp = 0
        for row in playoff_rows:

            playoff_temp = []
            tag = row.find_all("td")
            line = [v.get_text().strip().replace(",", "") for v in tag]

            try:
                playoff_datetemp.append(line[1])

                for i in range(2, 5):
                    playoff_temp.append(line[i])
            except:
                continue
            for i in range(5, len(line)):
                try:
                    playoff_temp.append(float(line[i]))
                except:
                    playoff_temp.append(np.NaN)

            playoff_temp = np.array(playoff_temp)
            playoff_temp = np.resize(playoff_temp, (1, n))

            if playoff_temp[0][0] != 0:
                playoff_frame.append(playoff_temp[0])
            tmp += 1
        playoff_frame = pd.DataFrame(playoff_frame, columns=tags, index=playoff_datetemp)
        playoff_frame.drop(columns='FILLER', inplace=True)
        dfs.append(playoff_frame)
        assert playoff_frame.shape[1] == frame.shape[1]
        dfs = pd.concat(dfs, axis = 0)
        dfs.index = frame.index.tolist() + playoff_frame.index.tolist()
        return dfs


def scrape_advanced(url):
    tags = ['Location', 'Opp', 'W/L', 'Tm_score', 'Opp_score',
            'ortg', 'drtg', 'pace', 'ftr', '3par', 'ts%', 'trb%', 'ast%', 'stl%',
            'blk%', 'FILLER', 'efg%_off', 'tov%_off', 'orb%_off', 'ft_fga_off', 'FILLER',
            'efg%_def', 'tov%_def', 'drb%_def', 'ft_fga_def']
    n = len(tags)

    r = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(r, 'html.parser')
    table = soup.find("table")

    table_body = table.find('tbody')
    rows = table_body.find_all('tr')

    frame = []
    datetemp = []
    tmp = 0
    dfs = []
    for row in rows:

        temp = []
        line = row.get_text().strip().split()
        tag = row.find_all("td")
        line = [v.get_text().strip().replace(",", "") for v in tag]

        try:
            datetemp.append(line[1])

            for i in range(2, 5):
                temp.append(line[i])
        except:
            continue
        for i in range(5, len(line)):
            try:
                temp.append(float(line[i]))
            except:
                temp.append(np.NaN)

        temp = np.array(temp)
        temp = np.resize(temp, (1, n))

        if temp[0][0] != 0:
            frame.append(temp[0])
        tmp += 1
    frame = pd.DataFrame(frame, columns=tags, index=datetemp)
    frame.drop(columns='FILLER', inplace=True)
    dfs.append(frame)

    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    comments = soup.find_all(string = lambda text: isinstance(text, Comment))
    tables = [comment for comment in comments if 'table_outer_container' in str(comment)]
    if len(tables) == 0:
        return frame
    else:
        comment = tables[0]
        table = BeautifulSoup(str(comment), 'html.parser').find('table').find('tbody')
        playoff_rows = table.find_all('tr')

        playoff_frame = []
        playoff_datetemp = []
        playoff_tmp = 0
        for row in playoff_rows:

            playoff_temp = []
            tag = row.find_all("td")
            line = [v.get_text().strip().replace(",", "") for v in tag]

            try:
                playoff_datetemp.append(line[1])

                for i in range(2, 5):
                    playoff_temp.append(line[i])
            except:
                continue
            for i in range(5, len(line)):
                try:
                    playoff_temp.append(float(line[i]))
                except:
                    playoff_temp.append(np.NaN)

            playoff_temp = np.array(playoff_temp)
            playoff_temp = np.resize(playoff_temp, (1, n))

            if playoff_temp[0][0] != 0:
                playoff_frame.append(playoff_temp[0])
            tmp += 1
        playoff_frame = pd.DataFrame(playoff_frame, columns=tags, index=playoff_datetemp)
        playoff_frame.drop(columns='FILLER', inplace=True)
        dfs.append(playoff_frame)
        assert playoff_frame.shape[1] == frame.shape[1]
        dfs = pd.concat(dfs, axis = 0)
        dfs.index = frame.index.tolist() + playoff_frame.index.tolist()
        return dfs

folder = pathlib.Path.cwd().parent / 'data' / 'Basketball' / 'Team' / 'gamelog'

for i, team in enumerate(teams):
    if i >= 7:
        print(f'Scraping For {team}')
        for year in range(first_year, last_year + 1):
            if team == 'BRK' and year < 2013:
                team_aka = 'NJN'
            elif team == 'CHO' and year < 2015 and year >= 2005:
                team_aka = 'CHA'
            elif team == 'CHO' and year < 2005:
                team_aka = 'CHH'
            elif team == 'MEM' and year < 2002:
                team_aka = 'VAN'
            elif team == 'NOP' and year >= 2006 and year < 2008:
                team_aka = 'NOK'
            elif team == 'NOP' and year < 2014:
                team_aka = 'NOH'
            elif team == 'OKC' and year < 2009:
                team_aka = 'SEA'
            elif team == 'SAC' and year < 1986:
                team_aka = 'KCK'
            elif team == 'WAS' and year < 1998:
                team_aka = 'WSB'
            else:
                team_aka = team
            url = url1 + team_aka + '/' + str(year) + url2
            advanced_url = url1 + team_aka + '/' + str(year) + '/gamelog-advanced/'
            try:
                time.sleep(10)
                temp = Scrap(url)
                temp = temp.reset_index(drop = False)
                advanced = scrape_advanced(advanced_url)
                advanced = advanced.reset_index(drop = False)
                merged_df = temp.merge(advanced, how='inner',
                                       left_on=['index', 'Location', 'Opp', 'W/L', 'Tm_score', 'Opp_score'],
                                       right_on=['index', 'Location', 'Opp', 'W/L', 'Tm_score', 'Opp_score'])
                merged_df = merged_df.set_index(['index'])
                dir = str(folder) + '/' + team + '_' + str(year) + '.csv'
                merged_df.to_csv(dir)
                print(str(year))
            except Exception as e:
                print(f'Error for {team} on {year}: {e}')
                continue

url1 + teams[0] + '/' + str(2007) + url2

