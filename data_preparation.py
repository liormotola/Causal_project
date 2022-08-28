import pandas as pd
import ast
from requests import get
from bs4 import BeautifulSoup
from datetime import datetime
from re import sub


def create_name(row):
    name = row['rotten_tomatoes_link']
    try:
        name = name.split('-')[1]
    except IndexError:
        name = name.split('/')[1]
    name = name.split('_')
    name = [i.lower() for i in name if isinstance(i, str)]
    return ' '.join(name)


def create_critics_data(begin_data, end_date):
    critics_data = pd.read_csv('data/rotten_tomatoes_critic_reviews.csv')
    critics_data = critics_data.dropna()
    critics_data['review_date'] = pd.to_datetime(critics_data['review_date'], infer_datetime_format=True)
    critics_data = critics_data[
        (critics_data['review_date'] >= begin_data) & (critics_data['review_date'] <= end_date)]
    critics_data['title'] = critics_data.apply(create_name, axis=1)
    critics_data = transform_scores(critics_data)
    return critics_data


def change_format(row):
    row = row['release_date'].split('-')
    return '/'.join(row)


def process(row):
    d = ast.literal_eval(row)
    temp = []
    for dct in d:
        temp.append(dct['name'])
    return "/".join(temp)

def transform_genres(data):
    data = data.apply(process)
    return data

def transform_to_workable_set(row):
    x = set()
    for info in ast.literal_eval(row):
        x.add(info['name'])
    return x


def find_top_k_companies_in_data_and_transform(data, k=10):
    temp = data.groupby('production_company').count().reset_index()
    relevant_ = temp[temp['movie_title'] > 1]['production_company'].values
    data = data.loc[
        [index for index in data.index if
         data.loc[index]['production_company'] in relevant_]]
    generic_words = ['Classics', 'Cinema', 'filmproduktion', 'eye', 'and', 'company', 'Distribution', 'Co.', 'Big',
                     'Picture',
                     'Group', 'Video', 'Films', 'New', 'Animation', 'Releasing', 'Screen', 'First', 'USA',
                     'Corporation', 'Motion'
                                    'New', 'Arts', 'The', 'Inc.', 'International', 'Home', 'Film', 'Films',
                     'Entertainment', 'Entertianment',
                     'Pictures', 'Studios', 'Releases', 'Services', 'Media', 'Productions', 'LLC.', 'Attractions',
                     'Premiere',
                     'Glass', 'hill', '&', 'Digital', 'Classic', ',', 'Go', 'Dark', 'sky', 'Independent', 'Features',
                     'Atrractions', 'High', 'Channel'
                                            'Factory', 'Vision', 'Street', 'Road', 'Box', 'Ventures', 'Motion',
                     'Brothers', 'Guild', 'World', 'Industries', 'Focus'
        , 'channel', 'crown', 'red', 'point', 'gems', 'line', 'blue', 'factory', 'national', 'roadside', 'corp.',
                     'american', 'music']
    generic_words = set([i.lower() for i in generic_words])
    prod_companies = list(set(data['production_company']))
    same_companies = {}
    remaining_companies = prod_companies.copy()
    for company in remaining_companies.copy():
        company = company.strip()
        for company_2 in remaining_companies.copy():
            company_2 = company_2.strip()
            company_name_intersection = set(company.split(' ')).intersection(set(company_2.split(' ')))
            if company == company_2 or len(company_name_intersection) == 0:
                continue
            company_name_intersection = set([i.lower() for i in company_name_intersection])
            if company_name_intersection.issubset(generic_words) or '/' in company or '/' in company_2:
                continue
            if company in remaining_companies:
                remaining_companies.remove(company)
            if company_2 in remaining_companies:
                remaining_companies.remove(company_2)
            if company in same_companies.keys():
                same_companies[company].append(company_2)
            else:
                same_companies[company] = [company_2]
    final_companies = {}
    for key in same_companies:
        x = [set(key.replace('-', ' ').split(' '))]
        for i in same_companies[key]:
            x.append(set(i.replace('-', ' ').split(' ')))
        final_name = set.intersection(*x)
        if len(final_name) > 0:
            final_name = " ".join(list(final_name))
            final_companies[final_name] = {key}.union(same_companies[key])
    for key in final_companies:
        for key2 in final_companies[key]:
            data.loc[
                data['production_company'] == key2, 'production_company'] = key
    final = data.groupby('production_company').count()['movie_title'].reset_index().rename(
        {'movie_title': 'count'}, axis=1)
    top_companies = list(final.nlargest(k, 'count')['production_company'].values)
    data['is_top_production_company'] = data.apply(lambda row:row['production_company'] in top_companies,axis = 1)
    return data


def create_movies_data(begin_data, end_date):
    movies_data_imdb = pd.read_csv('data/movies_metadata.csv').drop(['belongs_to_collection', 'homepage', 'tagline'], axis=1)
    movies_data_imdb = movies_data_imdb.dropna()
    movies_data_imdb['budget'] = movies_data_imdb['budget'].astype(int)
    movies_data_imdb = movies_data_imdb[movies_data_imdb['budget'] > 0]
    movies_data_imdb['revenue'] = movies_data_imdb['revenue'].astype(int)
    movies_data_imdb = movies_data_imdb[movies_data_imdb['revenue'] > 0]
    movies_data_imdb['release_date'] = movies_data_imdb.apply(change_format, axis=1)
    movies_data_imdb['release_date'] = pd.to_datetime(movies_data_imdb['release_date'], infer_datetime_format=True)
    movies_data_imdb = movies_data_imdb[
        (movies_data_imdb['release_date'] >= begin_data) & (movies_data_imdb['release_date'] <= end_date)]
    movies_data_imdb['title'] = movies_data_imdb.apply(lambda row: row['title'].lower(), axis=1)
    movies_data_imdb = movies_data_imdb[movies_data_imdb['original_language'] == 'en']
    movies_data_imdb = movies_data_imdb.drop(['original_title', 'original_language'], axis=1)
    movies_data_rotten_tomatoes = pd.read_csv('data/rotten_tomatoes_movies.csv')[
        ['movie_title', 'directors', 'actors', 'production_company', 'original_release_date']]
    movies_data_rotten_tomatoes = movies_data_rotten_tomatoes.dropna()
    movies_data_rotten_tomatoes['original_release_date'] = pd.to_datetime(
        movies_data_rotten_tomatoes['original_release_date'])
    movies_data_rotten_tomatoes = movies_data_rotten_tomatoes[
        (movies_data_rotten_tomatoes['original_release_date'] >= begin_data) & (movies_data_rotten_tomatoes['original_release_date'] <= end_date)]
    movies_data_rotten_tomatoes['production_company'] = movies_data_rotten_tomatoes['production_company'].apply(
        lambda row: row.lower())
    movies_data_rotten_tomatoes = find_top_k_companies_in_data_and_transform(movies_data_rotten_tomatoes)
    movies_data_rotten_tomatoes['title'] = movies_data_rotten_tomatoes['movie_title'].apply(str.lower)
    movies_data_rotten_tomatoes = movies_data_rotten_tomatoes.drop(['movie_title'], axis=1)

    data = pd.merge(movies_data_imdb, movies_data_rotten_tomatoes, on=['title'])
    data['diff'] = data['original_release_date'] - data['release_date']
    data['diff'] = data['diff'].apply(lambda x: abs(x.days))
    data = data[data['diff'] <= 31]
    data['release_date'] = data.apply(lambda row: min(row['original_release_date'], row['release_date']), axis=1)
    data = data.drop(['original_release_date','diff'], axis=1)
    data['genres'] = transform_genres(data['genres'])
    return data


def is_relevant(row):
    for letter in ['A', 'B', 'C', 'D', 'F']:
        if letter in row:
            return False
    if '/' not in row:
        return False
    return True


def change_grade_format(score):
    x = score.split('/')
    score = float(x[0]) / float(x[1]) * 10
    return score


def transform_scores(data):
    data = data[[is_relevant(i) for i in data['review_score']]]
    data['review_score'] = data['review_score'].apply(change_grade_format)
    return data


def get_date(tag):
    try:
        str_date = tag.find('td').text
        date = datetime.strptime(str_date, "%b %d, %Y")
        return date
    except:
        return None


def get_world_wide_box_office(tag):
    try:
        return int(sub(r'[^\d.]', '', tag.find_all('td')[-1].text))
    except:
        return None


def get_num_of_known_actors(row_data, actors_data):
    movie_year = row_data['release_date'].year
    actors = set(row_data['actors'].split(','))
    count = len(actors.intersection(actors_data[movie_year]))
    return count


def convert_actors(data):
    url_dict = {
        2010: "https://www.the-numbers.com/box-office-star-records/worldwide/yearly-acting/highest-grossing-2009-stars",
        2011: "https://www.the-numbers.com/box-office-star-records/worldwide/yearly-acting/highest-grossing-2010-stars",
        2012: "https://www.the-numbers.com/box-office-star-records/worldwide/yearly-acting/highest-grossing-2011-stars",
        2013: "https://www.the-numbers.com/box-office-star-records/worldwide/yearly-acting/highest-grossing-2012-stars",
        2014: "https://www.the-numbers.com/box-office-star-records/worldwide/yearly-acting/highest-grossing-2013-stars",
        2015: "https://www.the-numbers.com/box-office-star-records/worldwide/yearly-acting/highest-grossing-2014-stars",
        2016: "https://www.the-numbers.com/box-office-star-records/worldwide/yearly-acting/highest-grossing-2015-stars"}
    actors_data = {}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}
    for year, url in url_dict.items():
        actors_data[year] = set()
        response = get(url, headers=headers)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        container = html_soup.find("tbody").find_all("a")
        for actor in container:
            actors_data[year].add(actor.find(text=True))
    data['known_actors'] = data.apply(lambda row: get_num_of_known_actors(row, actors_data), axis=1)
    return data


def get_directors(row, top_directors):
    if len(set(row['directors'].split(',')).intersection(top_directors)) > 0:
        return True
    return False


def convert_directors(data_):
    directors_urls = ["https://www.imdb.com/list/ls056848274/",
                      "https://www.imdb.com/list/ls056848274/?sort=list_order,asc&mode=detail&page=2",
                      "https://www.imdb.com/list/ls056848274/?sort=list_order,asc&mode=detail&page=3"]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36',
    }
    top_directors = set()
    for url in directors_urls:
        response = get(url, headers=headers)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        container = html_soup.find("div", class_="lister-list").find_all("div", class_="lister-item mode-detail")
        for director in container:
            director = director.find("h3", class_="lister-item-header").text.split('.')[1].strip()
            top_directors.add(director)
    data_['known_directors'] = data_.apply(lambda row: get_directors(row, top_directors), axis=1)
    return data_


def main():
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2016, 12, 31)
    critics_data = create_critics_data(start_date, end_date)
    movies_data = create_movies_data(start_date, end_date)
    critics_data = movies_data.merge(critics_data, on='title')
    critics_data = critics_data[critics_data['release_date'] > critics_data['review_date']]
    critics_data = critics_data.groupby("title").mean().reset_index()[['title', 'review_score']]
    movies_data = convert_directors(movies_data)
    movies_data = convert_actors(movies_data)
    data = critics_data.merge(movies_data, on=['title'])
    cols_to_save = ['release_date', 'budget', 'review_score', 'is_top_production_company', 'known_actors', 'runtime',
                    'revenue', 'known_directors', 'popularity', 'title','genres']
    data = data[cols_to_save]
    data.to_csv("processed_data.csv", index=False)


if __name__ == '__main__':
    main()
