from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import requests
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import selenium
from selenium.webdriver.support import expected_conditions as EC
import pickle
import urllib
from collections import defaultdict
import pathlib, argparse, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


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

def download_url_data(driver, odds_url, date, match):
    # initial_url = url.split('basketball')[0]
    mapping = {0: 'payout',
               1: 'Over',
               2: 'Under'}
    # data = defaultdict(list)
    # odds_url = initial_url + link
    # html_doc = urllib.request.urlopen(odds_url).read()
    driver.get(odds_url)
    new_url = driver.current_url
    # driver.find_element_by_xpath('//*[@title="Over/Under"]').click()
    WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//*[@title='Over/Under']"))).click()
    # WebDriverWait(driver, 20).until(EC.element_to_be_clickable(
    #     (By.XPATH, '//*[@title="Over/Under"]'))
    # ).click()

    iter, max_iters, table_header = 0, 20, []
    data = defaultdict(list)
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

    return data

def scrape_tables(driver, ele1, url, log_dir, use_request, page_number, scrape = False):
    # Do more scraping
    # if use_request:
    #     soup_level2 = BeautifulSoup(page_source, 'html.parser')
    # else:
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
    if not scrape:
        return links
    else:
        all_data = {}

        for key, link in links.items():
            date, match = key
            data = download_url_data(driver, url, link, date, match)
            all_data[(date, match)] = data

        txt = '_'.join(ele1.text.split('/'))
        fname = f'{str(log_dir)}' + f'/{txt}_{page_number}.pkl'
        save_obj(fname, all_data)
        return


def scrape_tables_parallel(urls):
    # Do more scraping
    # if use_request:
    #     soup_level2 = BeautifulSoup(page_source, 'html.parser')
    # else:
    try:
        use_request = False
        scrape = True
        use_request = False
        key, url = urls
        initial_url = 'https://www.oddsportal.com/basketball/usa/nba/results/'.split('basketball')[0]
        url = initial_url + url
        if use_request:
            page_source = requests.get(url)
            soup_level2 = BeautifulSoup(page_source.content, 'html.parser')

        else:
            chromeOptions = webdriver.ChromeOptions()
            chromeOptions.add_argument("--headless")
            driver = webdriver.Chrome(chrome_options = chromeOptions)
            driver.get(url)
            time.sleep(0.25)
            soup_level2 = BeautifulSoup(driver.page_source, 'html.parser')

        # get data divider indices
        if not scrape:

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
            return links
        else:
            all_data = {}

            date, match = key
            data = download_url_data(driver, url, date, match)
            all_data[(date, match)] = data
            driver.quit()
            #
            # txt = '_'.join(ele1.text.split('/'))
            # fname = f'{str(log_dir)}' + f'/{txt}_{page_number}.pkl'
            # save_obj(fname, all_data)
            return all_data
    except Exception as e:
        print(f'Exception {e}')
        driver.quit()
        return {}

def scrape_pages(driver, ele1, all_pages, log_dir, url_reg, next_link, scrape = False, use_request = False):
    # loop over rest of pages
    error_url = []
    all_links = {}
    for page_number, next_page in enumerate(all_pages):
        url_append = ele1.find('a')['href']
        url = 'https://www.oddsportal.com/' + url_append + next_page
        driver.get(url)
        txt = '_'.join(ele1.text.split('/'))
        filename = f'{str(log_dir)}' + f'/{txt}_{page_number}.pkl'
        if pathlib.Path(filename).is_file() and (scrape == True):
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

                    element.send_keys(selenium.webdriver.common.keys.Keys.SPACE)

                    # driver.find_element_by_link_text(next_link.text).click()
                    if scrape:
                        scrape_tables(driver, ele1, url, log_dir, use_request, page_number, scrape=True)
                        time.sleep(0.25)
                    else:
                        links = scrape_tables(driver, ele1, url, log_dir, use_request, page_number, scrape=False)
                        all_links.update(links)
                        time.sleep(0.25)
                except Exception as e:
                    error_url.append(current_url)
                    print(f'Error on link : {current_url} as : {e}')
    if not scrape:
        return all_links
    return

def get_urls(scrape = True):
    chromeOptions = webdriver.ChromeOptions()
    chromeOptions.add_argument("--headless")
    driver = webdriver.Chrome(chrome_options = chromeOptions)
    url = 'https://www.oddsportal.com/basketball/usa/nba/results/'
    driver.get(url)
    driver.implicitly_wait(3)

    soup = BeautifulSoup(driver.page_source, 'lxml')

    phrase = re.compile(r'.*([0-9][0-9][0-9][0-9])')
    years = soup.find_all('', {'class': ['active', 'inactive']}, text=phrase)

    date_reg = re.compile(r'12B')
    result = []
    error_year = []
    error_url = []

    log_dir = pathlib.Path.cwd().parent / 'data' / 'odds_data_over_under'
    log_dir.mkdir(parents=True, exist_ok=True)
    all_links_years = {}
    for ele1 in years:
        try:
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
            all_link = scrape_pages(driver, ele1, all_pages, log_dir, url_reg, next_link, scrape=scrape, use_request=False)
            all_links_years.update(all_link)
        except Exception as e:
            print(f'Ran into error for {ele1}: {e}')
    driver.quit()
    return all_links_years

def main(args, urls):
    if args.parallel and args.scrape_data:
        maxworkers = 50
        threads = ThreadPoolExecutor(max_workers = maxworkers)
        # threads = ProcessPoolExecutor(max_workers = 8)
        result = list(threads.map(scrape_tables_parallel, urls))
        threads.shutdown()
        return result
    else:
        results = []
        for url in urls:
            result = scrape_tables_parallel(url)
            results.append(result)
        return results

def get_errors(inputs, result):
    check, errors = [], []
    for res in result:
        for k, v in res.items():
            if len(v) != 0:
                check += [(k, v)]
            else:
                errors += [(k, v)]

    s = set(map(lambda x: x[0], inputs))
    errors = s.difference(set(map(lambda x: x[0], check)))
    return errors

def convert_to_df(result):
    results = {}
    for res in result:
        for k, v in res.items():
            results[k] = pd.DataFrame.from_dict(v)
    return results

def parse_args():
    parser = argparse.ArgumentParser('webscraper for over under odds')
    parser.add_argument('-scrape_data', default = 'data', type = str, help = 'option: data vs urls')
    parser.add_argument('-add_timer', default = '1', type = int)
    parser.add_argument('-url_filename', default = 'odds_data_OU_urls_v2.pkl', type = str)
    parser.add_argument('-parallel', default = '1', type = int)
    parser.add_argument('-load_last_results', default = '0', type = int, help = 'load previous result and continue from last result')
    parser.add_argument('-load_all_results', default = '1', type = int)
    parser.add_argument('-filename', default = 'results.pkl', type = str)
    parser.add_argument('-dataframe_fname', default = 'odds_data_over_under_v2.feather', type = str)
    parser.add_argument('-process_pandas', default = '0', type = int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.scrape_data == 'data':
        directory = pathlib.Path.cwd().parent / 'data' / args.url_filename
        urls = load_obj(str(directory))
        inputs = list(urls.items())
        # threads = 13.579532146453857

        # get_urls(args.scrape_data)
        if args.load_last_results:
            fname_load = pathlib.Path.cwd().parent / 'data' / 'odds_data_over_under' / args.filename
            result = load_obj(str(fname_load))
            result_key = {}
            for res in result:
                if res is not None:
                    try:
                        result_key.update(res)
                    except:
                        continue
            inputs_new = [inp for inp in inputs if inp[0] not in result_key]
            inputs_tmp = []
            for i in range(0, len(inputs_new), 1000):
                print(f'Moving on to iteration {i // 1000}')
                inputs_tmp = inputs_new[i : i + 1000]
                inputs_save_name = pathlib.Path.cwd().parent / 'data' / 'odds_data_over_under' / f'inputs_v2_{i}.pkl'
                save_obj(str(inputs_save_name), inputs_tmp)
                print(f'Length of continued input is {len(inputs_tmp)}')
                result_new = main(args, inputs_tmp)
                result_success = [res for res in result_new if len(res) != 0]
                print(f'Number of success is {len(result_success)}')
                result_save_name = pathlib.Path.cwd().parent / 'data' / 'odds_data_over_under' / f'result_v2_{i}.pkl'
                save_obj(str(result_save_name), result_new)
        elif args.load_all_results:
            fname_all = pathlib.Path.cwd().parent / 'data' / 'odds_data_over_under'
            result = []
            names = [f'result_v2_{i * 1000}.pkl' for i in range(16)]
            for name in names:
                results_tmp = load_obj(str(fname_all / name))
                result += results_tmp
            print(f'Proportion of successful data scraped is {len(result) / len(inputs)}')

        else:
            t0 = time.time()
            result = main(args, inputs)
            t1 = time.time()
            print(f'Total time took {t1 - t0}')
            print(f'Len of result is {len(result)}')
            fname = pathlib.Path.cwd().parent / 'data' / 'odds_data_over_under' / 'results.pkl'
            save_obj(str(fname), result)
        # possible check for errors:
        # errors = get_errors(inputs, result)
        # new_results, i = result.copy(), 0
        # while len(errors) != 0 and i < 20:
        #     retries = [inp for inp in inputs if inp[0] in errors]
        #     retry_results = main(args, retries)
        #     new_results += retry_results
        #     errors = get_errors(retries, new_results)
        #     i += 1
    elif args.scrape_data == 'urls':
        directory = pathlib.Path.cwd().parent / 'data' / 'odds_data_OU_urls_v2.pkl'
        all_links_years = get_urls(args.scrape_data)
        save_obj(str(directory), all_links_years)

    if args.process_pandas:
        df_dict = convert_to_df(result)
        df = pd.concat(df_dict.values(), keys = df_dict.keys(), axis = 0)
        save_dir = pathlib.Path.cwd().parent / 'data' / 'odds_data_over_under' / 'odds_data_over_under_v2.feather'
        df = df.reset_index(drop = False)
        df = df.drop(['level_2'], axis = 1).rename({'level_0' : 'date', 'level_1' : 'match'}, axis = 1)
        df['match'] = df['match'].apply(lambda x: x.replace('\xa0', ''))
        df['date'] = df['date'].apply(lambda x: x.replace("12B's", ''))
        df = df.assign(team1 = lambda x: x['match'].apply(lambda x: x.split(' - ')[0]),
                       team2 = lambda x: x['match'].apply(lambda x: x.split(' - ')[1]))
        df['date'] = df['date'].apply(lambda x: x.split(' - ')[0])
        # cols = ['Over', 'Under']
        df = df.assign(under = lambda x: x['Over'], over = lambda x: x['Under'])
        df['over'] = pd.to_numeric(df['over'], errors = 'coerce')
        df['under'] = pd.to_numeric(df['under'], errors = 'coerce')
        df['US_Over'] = np.NaN
        df['US_Under'] = np.NaN
        df.loc[df['over'] >= 2.0, 'US_Over'] = (df.loc[df['over'] >= 2.0, 'over'] - 1) * 100
        df.loc[df['under'] >= 2.0, 'US_Under'] = (df.loc[df['under'] >= 2.0, 'under'] - 1)*100
        df.loc[df['over'] < 2.0, 'US_Over'] = (-100 / (df.loc[df['over'] < 2.0, 'over'] - 1))
        df.loc[df['under'] < 2.0, 'US_Under'] = (-100 / (df.loc[df['under'] < 2.0, 'under'] - 1))
        df.to_feather(str(save_dir))
    else:
        save_dir = pathlib.Path.cwd().parent / 'data' / 'odds_data_over_under' / 'odds_data_over_under_v2.feather'
        df = pd.read_feather(str(save_dir))
        count = df.groupby(['date', 'match']).ngroups

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


