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

if __name__ == '__main__':
    driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver')
    url = 'https://www.oddsportal.com/basketball/usa/nba/results/'
    driver.get(url)
    driver.implicitly_wait(3)

    use_request = False
    save = True

    soup = BeautifulSoup(driver.page_source, 'lxml')

    phrase = re.compile(r'.*([0-9][0-9][0-9][0-9])')
    years = soup.find_all('', {'class': ['active', 'inactive']}, text=phrase)

    date_reg = re.compile(r'12B')
    result = []
    error_year = []
    error_url = []
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
            next_link = soup_level2.find(name = None, class_ = 'arrow', text='»')
            next_url = next_link.parent.get('href')
            # Get data divider indices
            table = soup_level2.find('table', class_='table-main')

            pages = soup_level2.find('div', {'id' : 'pagination'})
            all_rows = table.find_all('tr')

            max_page = max([int(i.get('x-page')) for i in pages.find_all('a')])
            all_pages = ['#/page/{}/'.format(i) for i in range(2, max_page+1)]

            url_reg = re.compile(r'#/page/\d+/')

            # loop over rest of pages
            for next_page in all_pages:
                current_url = driver.current_url
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
                    except:
                        error_url.append(current_url)
                        print('Error on link : {}'.format(current_url))

                # Do more scraping
                if use_request:
                    soup_level2 = BeautifulSoup(page_source, 'html.parser')
                else:
                    soup_level2 = BeautifulSoup(driver.page_source, 'lxml')

                # get data divider indices
                table = soup_level2.find('table', class_='table-main')
                all_rows = table.find_all('tr')
                some_rows = table.find_all('tr', class_ = 'center nob-border')

                # get data
                table_rows = table.find_all('tr', {'class':['center nob-border', 'odd deactivate', 'deactivate']})

                col_names = ['Date', 'Time', 'Score', 'Odds1', 'Odds2', 'n_dealers']

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

    except:
        print('Error For Remaining Years, Stopping')

    correct_errors = []
    if len(error_url) != 0:
        for url in error_url:
            res = scrape_url(driver, url)
            correct_errors += res
    if save:
        save_path = './data/scraped_odds_data.pkl'
        save_obj(save_path, result)

    odds = process_result(result)
    df_odds = pd.DataFrame(odds, columns=['date', 'time', 'team1', 'team2', 'score', 'odds1', 'odds2', 'num_books'])

    if save:
        df_odds.to_csv('./data/scraped_odds_data.csv', index=False)


