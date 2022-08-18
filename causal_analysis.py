import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def resample_bins(data, treatment):
    new_data = []
    for label in data[treatment].unique():
        label_data = data[data[treatment] == label].reset_index(drop = True)
        n = len(label_data)
        bootstrap_indices = np.random.randint(n, size=n)
        label_bootstrap = label_data.iloc[bootstrap_indices]
        new_data.append(label_bootstrap)
    new_data = pd.concat(new_data)
    return new_data


def preprocess(path, score_bins, get_dummies=True):
    data = pd.read_csv(path)
    data['release_date'] = pd.to_datetime(data['release_date'])
    data['month'] = data.apply(lambda row: row['release_date'].month, axis=1)
    data['ROI'] = data['revenue'] / data['budget'] - 1
    data['release_date'] = pd.to_datetime(data['release_date'])
    data['revenue'] = np.log(data['revenue'])
    data['budget'] = np.log(data['budget'])
    data = bin_review_score(data, score_bins)
    if get_dummies:
        data = pd.get_dummies(data, columns=['month'])
        dummies = data['genres'].str.get_dummies(sep='/')
        genres_cols = list(dummies.columns)
        months_cols = [col for col in data.columns if 'month' in col]
        data = pd.concat([data, dummies], axis=1).drop(['genres'], axis=1)
        return data, genres_cols, months_cols
    else:
        return data


def transform_score(score, bins):
    for index, bin_ in enumerate(bins):
        if score >= bin_[0] and score <= bin_[1]:
            return index


def bin_review_score(data, k=2):
    quantile = 1 / k
    bins = [(0, data['review_score'].quantile(quantile))]
    for i in range(1, k - 1):
        bins.append((data['review_score'].quantile(quantile), data['review_score'].quantile(quantile + 1 / k)))
        quantile += 1 / k
    bins.append((data['review_score'].quantile(quantile), 10))
    data['score'] = data['review_score'].apply(lambda score: transform_score(score, bins))
    data = data.drop('review_score', axis=1)
    return data


def create_data_frames(data, continuous_features, categorical_features, treatment, labels_combination, trim):
    data_frames = pairwise_logistic_regression(data, continuous_features, categorical_features, treatment,
                                               labels_combination)
    if trim:
        trimmed_data_frames = {}
        for labels_pair in data_frames.keys():
            label_1, label_2 = labels_pair
            trimmed_data = trim_common_support_2_labels(data_frames[labels_pair], label_1, label_2)
            trimmed_data_frames[labels_pair] = trimmed_data.copy(deep=True)
        return trimmed_data_frames
    else:
        return data_frames


def pairwise_logistic_regression(data, continuous_features, categorical_features, treatment, labels_combinations):
    data_frames = {}
    for label_pair in labels_combinations:
        label_1, label_2 = label_pair
        curr_data = data[(data[treatment] == label_1) | (data[treatment] == label_2)]
        curr_data = curr_data.copy(deep=True)
        scaler = MinMaxScaler()
        curr_data[continuous_features] = scaler.fit_transform(curr_data[continuous_features])
        model = LogisticRegression('none', max_iter=1000)
        model.fit(curr_data[continuous_features + categorical_features], curr_data[treatment])
        propensity_scores = model.predict_proba(curr_data[continuous_features + categorical_features])
        curr_data[f'propensity_class_{label_1}'] = propensity_scores[:, 0]
        curr_data[f'propensity_class_{label_2}'] = propensity_scores[:, 1]
        data_frames[(label_1, label_2)] = curr_data
    return data_frames


def trim_common_support_2_labels(data, label_1, label_2):
    label_1_data = data[data['score'] == label_1]
    label_1_bounds = (
        label_1_data[f'propensity_class_{label_1}'].min(), label_1_data[f'propensity_class_{label_1}'].max())
    label_2_data = data[data['score'] == label_2]
    label_2_bounds = (
        label_2_data[f'propensity_class_{label_2}'].min(), label_2_data[f'propensity_class_{label_2}'].max())
    data = data[(data[f'propensity_class_{label_1}'] >= label_2_bounds[0]) & (
            data[f'propensity_class_{label_1}'] <= label_2_bounds[1])
                & (data[f'propensity_class_{label_2}'] >= label_1_bounds[0]) & (
                        data[f'propensity_class_{label_2}'] <= label_1_bounds[1])]
    return data


def plot_classes_propensity(data):
    for label in data['score'].unique():
        label_movies_propensity = data[data['score'] == label][f'propensity_class_{label}']
        plt.hist(label_movies_propensity, bins=20, label=label, alpha=0.5)
    plt.legend()
    plt.xlabel('propensity score')
    plt.ylabel('number of units')
    plt.xlim(0, 1)
    plt.show()


def add_interactions(data, treatment):
    for column in data.columns:
        if column in [treatment]:
            continue
        data[column + '_interaction'] = data[column].copy(deep=True) * data[treatment].copy(deep=True)
    return data


def generate_S_learner(data_frames, features, treatment, target):
    models = {}
    for labels_pair in data_frames.keys():
        data = data_frames[labels_pair]
        labels = data.pop(target)
        data = add_interactions(data[list(features) + [treatment]], treatment)
        model = Ridge(max_iter=100000)
        model.fit(data, labels)
        models[labels_pair] = model
    return models


def generate_T_learners(data_frames, features, treatment, target):
    models = {}
    for labels_pair in data_frames.keys():
        label_1, label_2 = labels_pair
        data = data_frames[labels_pair]
        data_label_1 = data[data[treatment] == label_1]
        model_1 = Ridge(max_iter=1000000)
        model_1.fit(data_label_1[features], data_label_1[target])
        data_label_2 = data[data[treatment] == label_2]
        model_2 = Ridge(max_iter=1000000)
        model_2.fit(data_label_2[features], data_label_2[target])
        models[labels_pair] = {}
        models[labels_pair][label_1] = model_1
        models[labels_pair][label_2] = model_2
    return models


def calc_all_ATE_T_learner(data_frames, features, treatment, target):
    models = generate_T_learners(data_frames, features, treatment, target)
    ATES = {}
    for labels_pair in data_frames.keys():
        label_1, label_2 = labels_pair
        data = data_frames[labels_pair]
        model_1 = models[labels_pair][label_1]
        model_2 = models[labels_pair][label_2]
        pred_1 = model_1.predict(data[features])
        pred_2 = model_2.predict(data[features])
        ATE = np.mean(pred_1 - pred_2)
        ATES[(label_1, label_2)] = ATE
    return ATES


def calc_all_ATE_S_learner(data_frames, features, treatment, target):
    models = generate_S_learner(data_frames, features, treatment, target)
    ATES = {}
    for labels_pair in data_frames.keys():
        label_1, label_2 = labels_pair
        data = data_frames[labels_pair]
        model = models[labels_pair]
        data_t_label_1 = data.copy(deep=True)
        data_t_label_1[treatment] = label_1
        data_t_label_2 = data.copy(deep=True)
        data_t_label_2[treatment] = label_2
        data_t_label_1 = add_interactions(data_t_label_1[features + [treatment]], treatment)
        data_t_label_2 = add_interactions(data_t_label_2[features + [treatment]], treatment)
        pred_1 = model.predict(data_t_label_1)
        pred_2 = model.predict(data_t_label_2)
        ATE = np.mean(pred_1 - pred_2)
        ATES[(label_1, label_2)] = ATE
    return ATES


def calculate_ATE_IPW(data_frames, target):
    ATES = {}
    for labels_pair in data_frames.keys():
        data = data_frames[labels_pair]
        ATE = 0
        label_1, label_2 = labels_pair
        label_1_data = data[data['score'] == label_1]
        propensity_label_1 = label_1_data.pop(f'propensity_class_{label_1}')
        target_label_1 = label_1_data.pop(target)
        label_2_data = data[data['score'] == label_2]
        target_label_2 = label_2_data.pop(f'propensity_class_{label_2}')
        n = len(data)
        ATE += 1 / n * sum(y / p for y, p in zip(target_label_1, propensity_label_1))
        ATE -= 1 / n * sum(y / (1 - p) for y, p in zip(target_label_2, propensity_label_1))
        ATES[labels_pair] = ATE
    return ATES


def continuous_distance(data_0, data_1, metric='euclidean'):
    scaler = MinMaxScaler()
    scaler.fit(np.vstack((data_0, data_1)))
    data_0 = scaler.transform(np.array(data_0))
    data_1 = scaler.transform(np.array(data_1))
    dist = pairwise_distances(data_0, data_1, metric=metric)
    return dist


def categorical_distance(data_0, data_1, metric='hamming'):
    data_0 = np.array(data_0)
    data_1 = np.array(data_1)
    dist = pairwise_distances(data_0, data_1, metric=metric)
    return dist


def create_dist(cat_dist, cont_dist):
    return np.multiply(cat_dist, cont_dist)


def calc_ATE_from_ITE(pairs_label_1, pairs_label_2, y_1, y_2):
    n = len(y_1) + len(y_2)
    total = sum(y_2[label_1] - y_1[label_2] for label_1, label_2 in pairs_label_1) / n
    total += sum(y_1[label_1] - y_2[label_2] for label_1, label_2 in pairs_label_2) / n
    return total


def calculate_ATE_matching(data_frames, categorical_features, continuous_features, treatment, target):
    ATES = {}
    for labels_pair in data_frames.keys():
        label_1, label_2 = labels_pair
        labels_data = data_frames[labels_pair]
        target_label_1 = np.array(labels_data[labels_data[treatment] == label_1][target])
        target_label_2 = np.array(labels_data[labels_data[treatment] == label_2][target])
        categorical_data_label_1 = labels_data[labels_data[treatment] == label_1][categorical_features]
        categorical_data_label_2 = labels_data[labels_data[treatment] == label_2][categorical_features]
        continuous_data_label_1 = labels_data[labels_data[treatment] == label_1][continuous_features]
        continuous_data_label_2 = labels_data[labels_data[treatment] == label_2][continuous_features]
        cont_dist = continuous_distance(continuous_data_label_1, continuous_data_label_2)
        cat_dist = categorical_distance(categorical_data_label_1, categorical_data_label_2)
        dist = create_dist(cat_dist, cont_dist)
        pairs_for_label_1 = [(i, neighbor) for i, neighbor in enumerate(np.argmin(dist, axis=0))]
        pairs_for_label_2 = [(i, neighbor) for i, neighbor in enumerate(np.argmin(dist, axis=1))]
        ATE = calc_ATE_from_ITE(pairs_for_label_1, pairs_for_label_2, target_label_1, target_label_2)
        ATES[labels_pair] = ATE
    return ATES


def main():
    continuous_features = ['budget', 'runtime']
    categorical_features = ['is_top_production_company', 'known_actors', 'known_directors']
    bins = 2
    original_data, genres, months = preprocess('processed_data.csv', bins, get_dummies=True)
    labels_combinations = [(1, 0)]
    bootstrap = 300
    results_s_learner = {k: [] for k in labels_combinations}
    results_t_learner = {k: [] for k in labels_combinations}
    results_ipw = {k: [] for k in labels_combinations}
    results_matching = {k: [] for k in labels_combinations}
    results = {'s_learner': results_s_learner, 't_learner': results_t_learner, 'ipw': results_ipw,
               'matching': results_matching}
    for i in tqdm(range(bootstrap)):
        data = resample_bins(original_data.copy(deep=True), treatment='score')
        features = categorical_features + continuous_features + genres + months
        data_frames = create_data_frames(data=data.copy(deep=True), continuous_features=continuous_features,
                                         categorical_features=categorical_features + genres + months, treatment='score',
                                         labels_combination=labels_combinations, trim=True)

        ATES = calc_all_ATE_S_learner(data_frames=data_frames, features=features, treatment='score', target='ROI')
        for key in ATES.keys():
            results_s_learner[key].append(ATES[key])
        data_frames = create_data_frames(data=data.copy(deep=True), continuous_features=continuous_features,
                                         categorical_features=categorical_features + genres + months, treatment='score',
                                         labels_combination=labels_combinations, trim=True)
        ATES = calc_all_ATE_T_learner(data_frames=data_frames, features=features, treatment='score', target='ROI')
        for key in ATES.keys():
            results_t_learner[key].append(ATES[key])
        data_frames = create_data_frames(data=data.copy(deep=True), continuous_features=continuous_features,
                                         categorical_features=categorical_features + genres + months, treatment='score',
                                         labels_combination=labels_combinations, trim=True)
        ATES = calculate_ATE_matching(data_frames=data_frames,
                                      categorical_features=categorical_features + months + genres,
                                      continuous_features=continuous_features
                                      , treatment='score', target='ROI')
        for key in ATES.keys():
            results_matching[key].append(ATES[key])
        data_frames = create_data_frames(data=data.copy(deep=True), continuous_features=continuous_features,
                                         categorical_features=categorical_features + genres + months, treatment='score',
                                         labels_combination=labels_combinations, trim=True)
        ATES = calculate_ATE_IPW(data_frames=data_frames, target='ROI')
        for key in ATES.keys():
            results_ipw[key].append(ATES[key])
    for learner in results.keys():
        for label_pair in results[learner].keys():
            results[learner][label_pair] = np.array(results[learner][label_pair])
            mean = results[learner][label_pair].mean()
            std = results[learner][label_pair].std()
            print(
                f"The result for the learner {learner} and the labels pair of {label_pair} is [{mean - std},{mean + std}]")


if __name__ == "__main__":
    main()
