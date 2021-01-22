from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import requests
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import urllib
from collections import defaultdict
import pathlib

def scrape_url(driver, current_url, use_request = False):
    url_reg = re.compile(r'#/page/\d+/')
    result = []
    # loop over rest of pages
    driver.get(current_url)
    print(current_url)

    if use_request:
        request = requests.get(current_url[:-1])
        page_source = request.text

    # Do more scraping
    if use_request:
        soup_level2 = BeautifulSoup(page_source, 'html.parser')
    else:
        soup_level2 = BeautifulSoup(driver.page_source, 'lxml')

    # get data divider indices
    table = soup_level2.find('table', class_='table-main')

    # get data
    table_rows = table.find_all('tr', {'class': ['center nob-border', 'odd deactivate', 'deactivate']})

    # create data col
    for i, tr in enumerate(table_rows):
        td = tr.find_all('td')
        row = [i.text for i in td]
        if len(row) == 0:
            date = tr.text
            if '-' in date:
                date = date.split('-')[0]
            elif '12B' in date:
                date = date.split('12B')[0]
            else:
                date = date
        else:
            row.insert(0, date)
            result.append(row)
    return result

def save_obj(path, object):
    with open(path,'wb') as file:
        pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as file:
        object = pickle.load(file)
    return object

def process_result(result_list):
    games = []
    for i, row in enumerate(result_list):
        team = row[2].split(' - ')
        team1 = team[0]
        team2 = team[1]
        if '\xa0' in team2:
            team2 = team2.replace('\xa0', '')
        row[2] = team1
        row.insert(3, team2)
        games.append(row)
    return games

def scrape_tables(driver, ele1, log_dir, use_request):
    # Do more scraping
    if use_request:
        soup_level2 = BeautifulSoup(page_source, 'html.parser')
    else:
        soup_level2 = BeautifulSoup(driver.page_source, 'html.parser')

    # get data divider indices
    table = soup_level2.find('table', class_='table-main')

    # get data
    table_rows = table.find_all('tr', {'class': ['center nob-border', 'odd deactivate', 'deactivate']})

    # create data col
    links = {}
    for i, tr in enumerate(table_rows):
        if ' '.join(tr['class']) == 'center nob-border':
            date = tr.text
        else:
            table_info = tr.find('td', {'class': 'name table-participant'})
            match = table_info.text
            link = table_info.contents[0]['href']
            key = (date, match)
            links[key] = link

    initial_url = url.split('basketball')[0]
    mapping = {0: 'payout',
               1: 'Over',
               2: 'Under'}

    all_data = {}

    for key, link in links.items():
        date, match = key
        data = defaultdict(list)
        odds_url = initial_url + link
        # html_doc = urllib.request.urlopen(odds_url).read()
        driver.get(odds_url)
        new_url = driver.current_url
        driver.find_element_by_xpath('//*[@title="Over/Under"]').click()
        iter, max_iters, table_header = 0, 20, []
        while (iter < max_iters) and (len(table_header) == 0):
            odds_soup = BeautifulSoup(driver.page_source, 'html.parser')
            table_header = odds_soup.find_all("div", class_="table-container")
            iter += 1

        for tag in table_header:
            if ('%' in tag.text) or ('(0)Compare odds' not in tag.text):
                handicap = tag.text.split(' ')[:2]
                if handicap[0] == 'Over/Under':
                    data[handicap[0]].append(handicap[1])
                    spans = tag.find_all('span')
                    for i, span in enumerate(spans):
                        if i in mapping:
                            data[mapping[i]] += [span.text]

        all_data[key] = data
    txt = '_'.join(ele1.text.split('/'))
    fname = f'{str(log_dir)}' + f'/{txt}_{page_number}.pkl'
    save_obj(fname, all_data)
    return

if __name__ == '__main__':
    driver = webdriver.Chrome()
    url = 'https://www.oddsportal.com/basketball/usa/nba/results/'
    driver.get(url)
    driver.implicitly_wait(3)

    use_request = False
    save = False

    soup = BeautifulSoup(driver.page_source, 'lxml')

    phrase = re.compile(r'.*([0-9][0-9][0-9][0-9])')
    years = soup.find_all('', {'class': ['active', 'inactive']}, text=phrase)

    date_reg = re.compile(r'12B')
    result = []
    error_year = []
    error_url = []

    log_dir = pathlib.Path.cwd().parent / 'data' / 'odds_data_over_under'
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        for ele1 in years:
            print('Scraping Year = {}'.format(ele1.text))
            driver.implicitly_wait(3)
            year_button = driver.find_element_by_link_text(ele1.text)
            try:
                year_button.click()
            except:
                print('Error For Year = {}'.format(ele1.text))
                error_year.append(ele1.text)
                continue
            driver.implicitly_wait(3)
            soup_level2 = BeautifulSoup(driver.page_source, 'lxml')
            # Get next link
            i, iter, next_link = 0, 20, None
            while (i < iter) and (next_link is None):
                soup_level2 = BeautifulSoup(driver.page_source, 'lxml')
                next_link = soup_level2.find(name = None, class_ = 'arrow', text='»')
                next_url = next_link.parent.get('href')
            # Get data divider indices
            table = soup_level2.find('table', class_='table-main')

            pages = soup_level2.find('div', {'id' : 'pagination'})
            all_rows = table.find_all('tr')

            max_page = max([int(i.get('x-page')) for i in pages.find_all('a')])
            all_pages = ['/#/'] + ['#/page/{}/'.format(i) for i in range(2, max_page+1)]

            url_reg = re.compile(r'#/page/\d+/')

            # loop over rest of pages
            for page_number, next_page in enumerate(all_pages):
                url = 'https://www.oddsportal.com/basketball/usa/nba/results/' + next_page
                driver.get(url)
                txt = '_'.join(ele1.text.split('/'))
                filename = f'{str(log_dir)}' + f'/{txt}_{page_number}.pkl'
                if pathlib.Path(filename).is_file():
                    print(f'{filename} exists, skipping...')
                    continue
                else:
                    current_url = driver.current_url
                    current_url = current_url.replace('//basketball', '/basketball')
                    matches = url_reg.findall(current_url)
                    if len(matches) != 0:
                        current_url = url_reg.sub(next_page, current_url)
                        driver.get(current_url)
                    print(current_url)

                    if use_request:
                        request = requests.get(current_url[:-1])
                        page_source = request.text

                    else:
                        try:
                            element = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.LINK_TEXT, next_link.text))
                            )
                            driver.find_element_by_link_text(next_link.text).click()
                            scrape_tables(driver, ele1, log_dir, use_request)
                        except Exception as e:
                            error_url.append(current_url)
                            print(f'Error on link : {current_url} as : {e}')


    except Exception as e:
        print(f'Ran into error for {ele1}: {e}')

    # correct_errors = []
    # if len(error_url) != 0:
    #     for url in error_url:
    #         res = scrape_url(driver, url)
    #         correct_errors += res
    # if save:
    #     save_path = './data/scraped_odds_data.pkl'
    #     save_obj(save_path, result)
    #
    # odds = process_result(result)
    # df_odds = pd.DataFrame(odds, columns=['date', 'time', 'team1', 'team2', 'score', 'odds1', 'odds2', 'num_books'])
    #
    # if save:
    #     df_odds.to_csv('./data/scraped_odds_data.csv', index=False)


