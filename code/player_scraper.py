#%%
import ray
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, sys
sys.path.append('./code/')
import argparse, pathlib
from webscraper import save_obj, load_obj

ray.init()
def download_data(url, name):
    html_doc = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    snap = soup.find('div', class_="table_outer_container")
    snap_body = snap.find('thead')
    column_tag = snap_body.find_all('th')
    column_name = []
    for item in column_tag:
        line = item.get_text().strip().split()
        if len(line) != 0:
            column_name.append(str(line[0]))
        elif (len(line) == 0) and (column_name[-1] == 'Tm'):
            column_name.append('at')
        else:
            column_name.append('outcome_for_Tm')
    snap_body = snap.find('tbody')
    rows = snap_body.find_all('tr')
    value_table = []
    for i, row in enumerate(rows):
        values_tag = row.find_all('td')
        line = [v.get_text().strip().replace(",", "") for v in values_tag]
        if len(line) != 0:
            value_array = [name] + list(map((lambda i: line[i]),range(1,len(line))))
            value_table.append(value_array)
    try:
        playoff_snap = soup.find('div', class_='placeholder').next_sibling.next_sibling
        newsoup = BeautifulSoup(playoff_snap, 'lxml')
        table = newsoup.table
        table_body = table.find('tbody').find_all('tr')
        for i, row in enumerate(table_body):
            values_tag = row.find_all('td')
            line = [v.get_text().strip().replace(",", "") for v in values_tag]
            if len(line) != 0:
                value_array = [name] + list(map((lambda i: line[i]),range(1,len(line))))
                value_table.append(value_array)
    except Exception as e:
        print(f'Error scraping Player {name} for Playoffs')
    data_table = pd.DataFrame(value_table,columns=['name']+column_name[2:])
    return data_table

def scrape_player_index(url):
    html_doc = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html_doc)
    header = ['player', 'link', 'year_from', 'year_to', 'pos', 'height','weight','birthdate','colleges']
    table_all = soup.find('div', {'id' : 'all_players'})
    table_sub = table_all.find('div', {'id' : 'div_players'})
    table = table_sub.find('table', {'id' : 'players'})
    rows = table.find('tbody').find_all('tr')
    frame = []
    for i, row in enumerate(rows):
        id_row = row.find('th').find('a')
        link = id_row.get('href').split('.')[0]
        name = id_row.text
        year_min = row.find('td', {'data-stat' : 'year_min'}).text
        year_max = row.find('td', {'data-stat' : 'year_max'}).text
        pos = row.find('td', {'data-stat':'pos'}).text
        height = row.find('td', {'data-stat':'height'}).text
        weight = row.find('td', {'data-stat' : 'weight'}).text
        bday = row.find('td', {'data-stat':'birth_date'}).text
        college = row.find('td', {'data-stat':'colleges'}).text
        all_row = [name, link, year_min, year_max, pos, height, weight, bday, college]
        frame.append(all_row)
    df = pd.DataFrame(frame, columns=header)
    return df

@ray.remote
def parallel_web_scraper(years, row):
    X_all = []
    errors_list = []
    for year in years:
        nametag = row['link']
        name = row['player']
        if year < row['year_from']:
            continue
        elif year > row['year_to']:
            continue
        else:
            url = 'https://www.basketball-reference.com/' + nametag + '/gamelog/' + str(year) + '/'
            try:
                X = download_data(url, name)
                X_all.append(X)
            except:
                print('Error For Player = {} at year = {}'.format(name, year))
                errors_list.append((name, year))
                continue
    try:
        X_df = pd.concat(X_all, axis = 0, ignore_index= True, sort = True)
    except:
        X_df = []
    return (name, X_df, errors_list)

def web_scraper(years, row):
    X_all = []
    errors_list = []
    for year in years:
        nametag = row['link']
        name = row['player']
        if year < row['year_from']:
            continue
        elif year > row['year_to']:
            continue
        else:
            url = 'https://www.basketball-reference.com/' + nametag + '/' + '/gamelog/' + str(year) + '/'
            advanced_url = url.replace('gamelog', 'gamelog-advanced')
            try:
                X = download_data(url, name)
                X_advanced = download_data(advanced_url, name)
                X_all.append(X)
            except Exception as e:
                print('Error For Player = {} at year = {}'.format(name, year))
                print(f'Error is {e}')
                errors_list.append((name, year))
                continue
    X_df = pd.concat(X_all, axis=0, ignore_index=True)
    return (name, X_df, errors_list)

def parse_args():
    parser = argparse.ArgumentParser(description = 'Scrape Player Data')
    parser.add_argument('-scrape_player_idx', type = int, default = '0', help = 'whether to scrape player index')
    parser.add_argument('-scrape_player_data', type = int, default = '1', help = 'scrape player data')
    parser.add_argument('-parallel', type = int, default = '0', help = 'run in parallel')
    parser.add_argument('-save', type = int, default = '1', help = 'save scraped data')
    parser.add_argument('-start_year', type = int, default = '2020', help = 'from year')
    parser.add_argument('-end_year', type = int, default = '2022', help = 'to year')
    return parser.parse_args()

#%%
if __name__ == '__main__':
    args = parse_args()
    scrape_player_idx = args.scrape_player_idx
    save = args.save
    parallel = args.parallel
    scrape_player_data = args.scrape_player_data
    print(args)
    if scrape_player_idx:
        base_url = 'https://www.basketball-reference.com/players/'
        letter = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        player_idx = {}
        for i, character in enumerate(letter):
            print('Scraping character = {}'.format(character))
            url = base_url + character + '/'
            try:
                df = scrape_player_index(url)
                player_idx[character] = df
            except:
                print('Error For Character = {}'.format(character))
                continue
        print('Done, saving to csv')
        player_index_df = pd.concat(list(player_idx.values()), names = list(player_idx.keys()), axis = 0, ignore_index=True)
        player_index_df.to_csv('../data/Basketball/players_index.csv', index = False)
    if scrape_player_data:
        letters = [chr(i) for i in range(ord('z'), ord('z') + 1)]
        # letters = ['j']
        for letter in letters:
            try:
                print('Processing ... {}'.format(letter))
                player_index = pd.read_csv('../data/Basketball/players_index.csv', header = 0)
                player_index = player_index.loc[(player_index.link.str.contains('/{}/'.format(letter))), :]
                if player_index.shape[0] != 0:
                    player_index['year_from'] = player_index['year_from'].astype(float)
                    player_index['year_to'] = player_index['year_to'].astype(float)
                    player_index = player_index.loc[(player_index['year_from'] >= 1985) | (player_index['year_to'] >= 1985), :]
                    years = [i for i in range(args.start_year, args.end_year)] # set years
                    result = []
                    if parallel:
                        for _, row in player_index.iterrows():
                            time.sleep(2)
                            result.append(parallel_web_scraper.remote(years, row))
                        result = ray.get(result)
                    else:
                        for _, row in player_index.iterrows():
                            # if row['link'] == '/players/j/jamesle01':
                            result.append(web_scraper(years, row))

                    errors_list_all = []
                    X_all = []
                    for i, item in enumerate(result):
                        name, X, error_list = item
                        if isinstance(X, pd.DataFrame):
                            X_all.append(X)
                        else:
                            errors_list_all.append(error_list)

                    errors_list_all = [i for i in errors_list_all if len(i) != 0]

                    df_X = pd.concat(X_all, ignore_index=True, axis = 0)

                    if save:
                        save_obj('../data/Basketball/Player/raw_player_data_letter_{}_{}.pkl'.format(args.start_year, letter), result)
                        save_obj('../data/Basketball/Player/player_data_errors_letter_{}_{}.pkl'.format(args.start_year, letter), errors_list_all)
                        df_X.to_csv('../data/Basketball/Player/players_data_letter_{}_{}.csv'.format(args.start_year, letter))
            except Exception as e:
                print(f'Error for {letter}: {e}')
                print(f'Skipping ... {letter}')

    path = '../data/Basketball/Player/players_data_letter_{}_{}.csv'
    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    df_orig = pathlib.Path.cwd().parent / 'data' / 'Basketball' / 'Player' / 'players_data.csv'
    dfs_prev = pd.read_csv(str(df_orig), header = 0, index_col = 0)
    dfs = [dfs_prev]
    years = [i for i in range(args.start_year, args.end_year)]
    for year in years:
        for letter in letters:
            try:
                if letter != 'x':
                    df = pd.read_csv(path.format(year, letter), header = 0, index_col= 0)
                    dfs.append(df)
            except:
                print(f'Skipping letter {letter}')
    dfs = pd.concat(dfs, axis = 0, ignore_index= True, sort = False)
    # dfs.to_csv('./data/Basketball/players_data.csv')
    df = pd.concat([dfs_prev, dfs], axis = 0, ignore_index = True)
    df = df.drop_duplicates(subset = ['Date', 'name', 'Opp', 'Tm'])
    df.to_csv('../data/Basketball/players_data.csv')