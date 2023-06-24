import logging
from datetime import datetime, timedelta

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import time
from selenium import webdriver
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import warnings

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
handle = "my-app"
logger = logging.getLogger("my-app")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}


def crawler():
    data = {"company_name": [], "company_year_founded": [], "company_employees": [], "company_hq": [], "company_annual_revenue": [], "industry": [], "company_total_visits": [], "company_bounce_rate": [],
            "company_pages_per_visit": [], "company_avg_visit_duration_in_seconds": [], "company_3_month_ago": [], "company_2_month_ago": [], "company_1_month_ago": [], "company_demographics_gender_female": [], "company_demographics_gender_male": [], "company_age_distribution_18_24": [], "company_age_distribution_25_34": [], "company_age_distribution_35_44": []}
    df = pd.DataFrame(data)
    categories_urls = get_categories_urls()

    category_index = 0
    for category_url in categories_urls:
        category_index += 1
        category_retry = 3
        while category_retry > 0:
            logger.info(f'category_index: {category_index}, category_url: {category_url}')
            try:
                websites_urls = get_websites_urls_from_category(category_url=category_url)
            except:
                category_retry -= 1
                continue
            website_index = 0
            for website_url in websites_urls:
                website_index += 1
                website_retry = 3
                while website_retry > 0:
                    try:
                        logger.info(f'website_index: {website_index}, website_url: {website_url}')
                        driver = webdriver.Chrome()
                        driver.get(website_url)
                        time.sleep(5)
                        page_source = driver.page_source
                        driver.quit()
                        soup = BeautifulSoup(page_source, 'html.parser')
                        company_details = soup.find_all('dd', class_='app-company-info__list-item app-company-info__list-item--value')
                        company_name = company_details[0].text.strip()
                        company_year_founded = company_details[1].text.strip()
                        company_employees = company_details[2].text.strip().split()[-1]
                        company_hq = company_details[3].text.strip()
                        company_annual_revenue = company_details[4].text.strip()[:-1].split('$')[-1]
                        industry = company_details[5].text.strip()
                        company_general_stats = soup.find_all('p', class_='engagement-list__item-value')
                        company_total_visits = company_general_stats[0].text.strip()[:-1]
                        company_bounce_rate = company_general_stats[1].text.strip()[:-1]
                        company_pages_per_visit = company_general_stats[2].text.strip()
                        company_avg_visit_duration = company_general_stats[3].text.strip()
                        x = time.strptime(company_avg_visit_duration, '%H:%M:%S')
                        company_avg_visit_duration_in_seconds = timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()

                        company_total_visits_last_3_months = soup.find_all('tspan', class_='wa-traffic__chart-data-label')
                        company_3_month_ago = company_total_visits_last_3_months[0].text.strip()[:-1]
                        company_2_month_ago = company_total_visits_last_3_months[1].text.strip()[:-1]
                        company_1_month_ago = company_total_visits_last_3_months[2].text.strip()[:-1]
                        company_demographics_gender_stats = soup.find_all('span', class_='wa-demographics__gender-legend-item-value')
                        company_demographics_gender_female = company_demographics_gender_stats[0].text.strip()[:-1]
                        company_demographics_gender_male = company_demographics_gender_stats[1].text.strip()[:-1]
                        company_age_distribution_stats = soup.find_all('tspan', class_='wa-demographics__age-data-label')
                        company_age_distribution_18_24 = company_age_distribution_stats[0].text.strip()[:-1]
                        company_age_distribution_25_34 = company_age_distribution_stats[1].text.strip()[:-1]
                        company_age_distribution_35_44 = company_age_distribution_stats[2].text.strip()[:-1]
                        new_data = {"company_name": company_name, "company_year_founded": company_year_founded, "company_employees": company_employees, "company_hq": company_hq, "company_annual_revenue": company_annual_revenue, "industry": industry, "company_total_visits": company_total_visits, "company_bounce_rate": company_bounce_rate,
                        "company_pages_per_visit": company_pages_per_visit, "company_avg_visit_duration_in_seconds": company_avg_visit_duration_in_seconds, "company_3_month_ago": company_3_month_ago, "company_2_month_ago": company_2_month_ago, "company_1_month_ago": company_1_month_ago, "company_demographics_gender_female": company_demographics_gender_female, "company_demographics_gender_male": company_demographics_gender_male, "company_age_distribution_18_24": company_age_distribution_18_24, "company_age_distribution_25_34": company_age_distribution_25_34, "company_age_distribution_35_44": company_age_distribution_35_44}
                        df.loc[len(df)] = new_data
                        break
                    except:
                        website_retry -= 1
                        continue

            df.to_csv(
                r'C:\Users\asus\PycharmProjects\simillarweb_data_science_project\SimilarwebProject.csv')
            break

    pass


def get_categories_urls():
    url = 'https://www.similarweb.com/category/'
    response = requests.get(url, headers=headers)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    category_links = soup.find_all('a', class_='tl-list__link')
    urls = ['https://www.similarweb.com' + link['href'] for link in category_links]
    return urls


def get_websites_urls_from_category(category_url):
    response = requests.get(category_url, headers=headers)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    websites_links = soup.find_all('a', class_='tw-table__row-compare')
    websites_urls = ['https://www.similarweb.com' + link['href'] + '#overview' for link in websites_links]
    return websites_urls


def clean_data():
    df = pd.read_csv(r'C:\Users\asus\PycharmProjects\simillarweb_data_science_project\SimilarwebProject.csv', index_col=0)
    # print(df.dtypes)
    df.drop_duplicates(subset=["company_name"], inplace=True, ignore_index=True)
    df.company_name = df.company_name.apply(lambda x: np.nan if x.count('-') > 0 else x)
    df.company_year_founded = df.company_year_founded.apply(lambda x: np.nan if x.count('-') > 0 else x)
    df.company_employees = df.company_employees.apply(lambda x: np.nan if x.count('-') > 0 else x)
    df.company_hq = df.company_hq.apply(lambda x: np.nan if x.count('-') > 0 else x)
    df.company_annual_revenue = df.company_annual_revenue.apply(lambda x: np.nan if x.count('-') > 0 else x)
    df.industry = df.industry.apply(lambda x: np.nan if x.count('-') > 0 else x)
    # df.company_total_visits = df.company_total_visits.apply(lambda x: np.nan if x.isnumeric() else x)
    df.company_bounce_rate = df.company_bounce_rate.apply(lambda x: np.nan if not isfloat(x) and not isint(x) else x)
    # df.company_pages_per_visit = df.company_pages_per_visit.apply(lambda x: np.nan if x.isnumeric() else x)
    # df.company_avg_visit_duration_in_seconds = df.company_avg_visit_duration_in_seconds.apply(lambda x: np.nan if x.isnumeric() else x)
    df.company_3_month_ago = df.company_3_month_ago.apply(
        lambda x: np.nan if not isfloat(x) and not isint(x) else x)
    df.company_2_month_ago = df.company_2_month_ago.apply(
        lambda x: np.nan if not isfloat(x) and not isint(x) else x)
    # df.company_1_month_ago = df.company_1_month_ago.apply(
    #     lambda x: np.nan if x.isnumeric() else x)
    df.company_demographics_gender_female = df.company_demographics_gender_female.apply(
        lambda x: np.nan if not isfloat(x) and not isint(x) else x)
    df.company_demographics_gender_male = df.company_demographics_gender_male.apply(
        lambda x: np.nan if not isfloat(x) and not isint(x) else x)
    df.company_age_distribution_18_24 = df.company_age_distribution_18_24.apply(
        lambda x: np.nan if not isfloat(x) and not isint(x) else x)
    df.company_age_distribution_25_34 = df.company_age_distribution_25_34.apply(
        lambda x: np.nan if not isfloat(x) and not isint(x) else x)
    df.company_age_distribution_35_44 = df.company_age_distribution_35_44.apply(
        lambda x: np.nan if not isfloat(x) and not isint(x) else x)





    # df.company_year_founded = df.company_year_founded.apply(lambda x: x.replace(r'[^a-zA-Z0-9]', ''))
    # df.company_employees = df.company_employees.apply(lambda x: x.replace(r'[^a-zA-Z0-9]', ''))
    df['company_name'] = df['company_name'].str.replace(r'[^a-zA-Z0-9]', '')
    df['company_hq'] = df['company_hq'].str.replace(r'[^a-zA-Z0-9]', '')
    df['industry'] = df['industry'].str.replace(r'[^a-zA-Z0-9]', '')
    # df['industry'] = df['industry'].str.replace(' ', '')
    # df['industry'] = df['industry'].str.lower()
    # df.company_annual_revenue = df.company_annual_revenual_revenue.apply(str(x).replace(r'[^a-zA-Z0-9]', ''))
    # df.dropna(inplace=True)
    df.dropna(inplace=True)
    # df_filtered = df[df['company_year_founded'] != '--']
    # indexCommon = df[ (df['company_year_founded']) == '-' | (df['company_year_founded'] == '--')].index
    # df.drop(indexCommon, inplace=True)
    # ) & ((df['company_employees'] == '-') | (df['company_employees'] == '--')) & ((df['company_hq'] == '-') | (df['company_hq'] == '--')) & ((df['company_annual_revenue']) == '-' | (df['company_annual_revenue'] == '--'))
    industry_counts = df['industry'].value_counts()
    single_industries = industry_counts[industry_counts == 1].index
    df = df[~df['industry'].isin(single_industries)]
    df.to_csv(r"C:\Users\asus\PycharmProjects\simillarweb_data_science_project\SimilarwebProject_cleaned.csv")


def eda():
    df = pd.read_csv(r'C:\Users\asus\PycharmProjects\simillarweb_data_science_project\SimilarwebProject_cleaned.csv', index_col=0)
    plt.figure(figsize=(12, 6))
    sns.countplot(x='industry', data=df)
    plt.show()


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def isint(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def ml():
    df = pd.read_csv(r'C:\Users\asus\PycharmProjects\simillarweb_data_science_project\SimilarwebProject_cleaned.csv',
                     index_col=0)
    categorical_columns = ['company_name', 'company_hq']
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    X = df.drop(columns=["industry"], axis=1)

    y = label_encoder.fit_transform(df['industry'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rd = RandomForestClassifier()
    lg = LogisticRegression()
    naive = MultinomialNB()
    svm = SVC(kernel='linear')
    best_recall_val = 0
    result = []
    for clf, model in zip([rd, naive, lg], ["Random Forest", "Naive Bayes", "Logistic Regression"]):
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        clf_score_test = accuracy_score(y_true=y_test, y_pred=y_predicted)
        result.append({"Model": model, "y_test_score": clf_score_test})
        print(f"\n\nThe model {model} have score of {clf_score_test}.\n Confusion Matrix:\n")
        cm = confusion_matrix(y_true=y_test, y_pred=y_predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
        disp.plot(include_values=True)
        plt.figure(figsize=(12, 6))
        # plt.show()
        print("\n\n")

    result_df = pd.DataFrame(result)
    result_df = result_df.set_index('Model')
    fig = plt.figure(figsize=(7, 7))  # Create matplotlib figure
    ax = fig.add_subplot(111)  # Create matplotlib axes
    width = .3
    result_df.y_test_score.plot(kind='bar', color='green', ax=ax, width=width, position=0)
    ax.set_ylabel('Test on y_test')
    print("\n")
    # plt.show()

    parameters = {'n_estimators': range(50, 550, 50)}
    rd = RandomForestClassifier()
    gs = GridSearchCV(rd, parameters)
    gs.fit(X_train, y_train)
    print(gs.best_params_)

    rd = RandomForestClassifier(**gs.best_params_)
    rd.fit(X_train, y_train)
    y_predicted = rd.predict(X_test)
    score = accuracy_score(y_true=y_test, y_pred=y_predicted)
    cm = confusion_matrix(y_true=y_test, y_pred=y_predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(include_values=True)
    plt.show()
    print('\n', score)


def main():
    clean_data()
    ml()


if __name__ == "__main__":
    main()
