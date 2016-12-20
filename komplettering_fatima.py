"""
Tillämpad Programmering, HT 2016
Fatima Guseinova
______________________________________
This program uses the folder enron3 in my Mumin-directory.
The source of the files can be easily changed in the main-part.

"""

import os, random, nltk
from nltk import word_tokenize, WordNetLemmatizer, NaiveBayesClassifier, classify
from nltk.corpus import stopwords
from collections import Counter
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

stopw_list = stopwords.words('english')


def create_lists(folder):  # reads the data
    """ reads the spam and ham in two separate lists that are created in the main-part of the program """

    list_of_files = os.listdir(folder)
    the_list = []
    for the_file in list_of_files:
        a = open(folder + the_file, mode='r', encoding='latin-1')
        the_list.append(a.read())
    a.close()
    return the_list


def prepare_data (sentence):  # normalising the data

    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(the_word.lower()) for the_word in word_tokenize(sentence)]


def define_feat(text, setting):  # defines the features for a spam email

    if setting == 'bow':  # based on word frequency
        return {the_word: count for the_word, count in Counter(prepare_data(text)).items() if not the_word in stopw_list}
    else:
        return {the_word: True for the_word in prepare_data(text) if not the_word in stopw_list}


def train(features, sample_groups):  # a division of data into sample groups, one for training and one for testing

    training_size = int(len(features) * sample_groups)

    # the training commences
    training_set, testing_set = features[:training_size], features[training_size:]

    print(str(len(training_set)) + ' emails are being used in the training data, and \n' +
          str(len(testing_set)) + ' emails are being used in the test data.\n')

    nb_classifier = NaiveBayesClassifier.train(training_set)  # trains the Naive Bayes classifier
    bernoulli_classifier = SklearnClassifier(BernoulliNB()).train(training_set)  # trains the Bernoulli classifier
    log_reg_classifier = SklearnClassifier(LogisticRegression()).train(
        training_set)  # trains the LogisticRegression classifier

    return training_set, testing_set, nb_classifier, bernoulli_classifier, log_reg_classifier


def evaluation(training_set, testing_set, nb_classifier, bernoulli_classifier, log_reg_classifier):
    # evaluates the classifiers based on the training and testing data

    nb_training_accuracy = classify.accuracy(nb_classifier, training_set)
    nb_test_accuracy = classify.accuracy(nb_classifier, testing_set)
    bernoulli_accuracy = classify.accuracy(bernoulli_classifier, testing_set)
    log_reg_accuracy = classify.accuracy(log_reg_classifier, testing_set)

    print('Training accuracy: ' + str(nb_training_accuracy) + '.\n' +
          'Test accuracy: ' + str(nb_test_accuracy) + '.\n\n')

    nb_classifier.show_most_informative_features(20)  # shows the 20 most revealing spam words

    print("\n\n\nNaive Bayes Classifier accuracy in percent: ", round(nb_test_accuracy * 100), '%.\n')
    print("Bernoulli Classifier accuracy in percent: ", round(bernoulli_accuracy * 100), '%.\n')
    print("LogisticRegression Classifier accuracy in percent: ", round(log_reg_accuracy * 100), '%.\n')

    print("\n                          There you go, I'm done!\n")


if __name__ == "__main__":

    print('Let me see...\n')

    spam = create_lists('enron3/spam/')
    ham = create_lists('enron3/ham/')

    emails = [(email, 'OK') for email in ham]
    emails += [(email, 'spam') for email in spam]
    random.shuffle(emails)

    all_features = [(define_feat(email, ' '), label) for (email, label) in emails]

    print('I have found ' + str(len(all_features)) + ' feature sets.\n')

    training_set, testing_set, nb_classifier, bernoulli_classifier, log_reg_classifier = train(
        all_features, 0.8)  # uses a division of 80% for training data and 20% for testing data

    evaluation(training_set, testing_set, nb_classifier, bernoulli_classifier, log_reg_classifier)