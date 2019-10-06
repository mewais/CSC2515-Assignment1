import random
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

training_annot = []
validation_annot = []
test_annot = []
training_vector = []
validation_vector = []
test_vector = []


def load_data(useHashing = False):
    global training_annot
    global validation_annot
    global test_annot
    global training_vector
    global validation_vector
    global test_vector
    # Open files
    fake_news_file = open('clean_fake.txt')
    real_news_file = open('clean_real.txt')
    # Read entire data in lines
    fake_news = fake_news_file.read().split('\n')
    real_news = real_news_file.read().split('\n')
    # Mixup
    all_news_annot = [True]*len(fake_news)
    all_news = fake_news
    all_news_annot.extend([False]*len(real_news))
    all_news.extend(real_news)
    # Shuffle and Split dataset
    combined = list(zip(all_news_annot, all_news))
    random.shuffle(combined)
    all_news_annot[:], all_news[:] = zip(*combined)
    training_annot = all_news_annot[:(len(all_news_annot)*7)//10]
    validation_annot = all_news_annot[(len(all_news_annot)*7)//10:(len(all_news_annot)*85)//100]
    test_annot = all_news_annot[(len(all_news_annot)*85)//100:]
    training = all_news[:(len(all_news)*7)//10]
    validation = all_news[(len(all_news)*7)//10:(len(all_news)*85)//100]
    test = all_news[(len(all_news)*85)//100:]
    # Initiate vectorizer
    if useHashing:
        vectorizer = HashingVectorizer(n_features=2**10)
    else:
        vectorizer = CountVectorizer()
    training_vector = vectorizer.fit_transform(training)
    validation_vector = vectorizer.fit_transform(validation)
    test_vector = vectorizer.fit_transform(test)


def train_model(maxDepth = 10, infoGain = True):
    if infoGain:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=maxDepth)
    else:
        clf = DecisionTreeClassifier(criterion='gini', max_depth=maxDepth)
    clf = clf.fit(training_vector, training_annot)
    return clf


def test_model(clf, validate = True):
    if validate:
        return clf.score(validation_vector, validation_annot)
    else:
        return clf.score(test_vector, test_annot)


def select_model():
    best = ('None', 0, 0, None)
    for i in [3, 4, 5, 6, 8, 10, 15, 20, 30, 50]:
        entropy_clf = train_model(i)
        entropy_acc = test_model(entropy_clf)
        if entropy_acc > best[2]:
            best = ('Entropy', i, entropy_acc, entropy_clf)
        gini_clf = train_model(i, False)
        gini_acc = test_model(gini_clf)
        if gini_acc > best[2]:
            best = ('Gini', i, gini_acc, gini_clf)
        print 'Entropy-' + str(i) + ': ' + str(entropy_acc*100) + '%'
        print 'Gini-' + str(i) + ': ' + str(gini_acc*100) + '%'
    print '\n\n'
    print 'Best configuration was: ' + best[0] + ' with a ' + str(best[1]) + ' level maximum depth'
    print 'It scored: ' + str(best[2]*100) + '% on the validation set, and ' + str(test_model(best[3], False)*100) + '% on the test set.'
    return best[3]


if __name__ == '__main__':
    load_data(True)
    best = select_model()
    # Display graph
    data = export_graphviz(best, max_depth=5, out_file=None)
    graph = graphviz.Source(data)
    graph.render('graph')


