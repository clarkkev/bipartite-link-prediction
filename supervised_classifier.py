import util
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingClassifier


def get_features(u, b, unsupervised_scores, user, business):
    features = {score_name: unsupervised_scores[score_name][u][b]
                for score_name in unsupervised_scores}
    features["business_stars"] = float(business["stars"])
    features["business_reviews"] = float(business["review_count"])
    features["user_reviews"] = float(user["review_count"])

    return features


def X_y_e(is_train, vectorizer):
    print "Loading data..."
    dataset = "train" if is_train else "test"
    examples = util.load_json('./data/' + dataset + '/examples.json')
    users = util.load_json('./data/' + dataset + '/user.json')
    businesses = util.load_json('./data/' + dataset + '/business.json')
    unsupervised_scores = {f: util.load_json('./data/' + dataset + '/' + f + '.json') for f in
                           ['svd', 'weighted_random_walks', 'random_walks', 'b_adamic', 'b_cn',
                            'b_jaccard', 'u_adamic', 'u_cn', 'u_jaccard']}

    print "Computing features..."
    feature_dicts, y, e = [], [], []
    for u in examples:
        for b in examples[u]:
            e.append((u, b))
            y.append(examples[u][b])
            feature_dicts.append(get_features(u, b, unsupervised_scores, users[u], businesses[b]))
    X = vectorizer.fit_transform(feature_dicts) if is_train else vectorizer.transform(feature_dicts)

    return X, y, e


def train_test(X_train, y_train, X_test, e_test, vectorizer):
    print "Training..."
    clf = GradientBoostingClassifier(n_estimators=2000, max_depth=4)
    clf.fit(X_train, y_train)

    print "Testing..."
    probas = clf.predict_proba(X_test)[:, 1]

    scores = defaultdict(dict)
    for (u, b), p in zip(e_test, probas):
        scores[u][b] = p
    util.write_json(scores, './data/test/supervised_classifier.json')


def main():
    vectorizer = DictVectorizer(sparse=False)
    X_train, y_train, e_train = X_y_e(True, vectorizer)
    X_test, y_test, e_test = X_y_e(False, vectorizer)
    train_test(X_train, y_train, X_test, e_test, vectorizer)


if __name__ == '__main__':
    main()