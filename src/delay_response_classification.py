
import pandas as pd
import re
import numpy as np
from statistics import mean
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
import gensim
from gensim.models import Word2Vec

def text_preprocessing(text_series):
    processed_texts = text_series
    processed_texts = processed_texts.str.lower()
    processed_texts = processed_texts.str.replace('[^a-zA-Z]', ' ')
    processed_texts = processed_texts.str.replace(r'\s+', ' ')

    return processed_texts

def compute_avg_vec(data, dim_size, model):
    '''
    Input:
        - data: list of lists where the element lists contain the words in each text.
    Output:
        - avg_vec_arr: the array of averaged vectors in a document.
        Each row is the average vector representations for a document.
    '''
    # compute average word representations
    avg_vec_arr = np.zeros((len(data), dim_size)).astype(float)
    for idx, update_tokens in enumerate(data):
        vectors_arr = np.zeros((len(update_tokens), dim_size))
        vectors_lst = []
        for word in update_tokens:
            try:
                word_vector = model.wv.word_vec(word)
                vectors_lst.append(word_vector.tolist())
            except:
                continue

        # average the vectors for words
        avg_vec = np.fromiter(map(mean, zip(*vectors_lst)), dtype=np.float)
        avg_vec_arr[idx,:] = avg_vec

    return avg_vec_arr

def evaluate(y_true, y_pred, y_pred_prob):
    # Accuracy
    eval_accuracy = accuracy_score(y_true, y_pred)

    # AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    eval_auc = auc(fpr, tpr)

    # F-1 score
    eval_f1 = f1_score(y_true, y_pred)

    return eval_accuracy, eval_auc, eval_f1


def adaboost_classification(train_x, train_y, test_x, test_y):
    ab_classifier = AdaBoostClassifier()
    ab_classifier.fit(train_x, train_y)
    pred = ab_classifier.predict(test_x)
    pred_prob = ab_classifier.predict_proba(test_x)
    pred_prob = pred_prob[:,np.where(ab_classifier.classes_==1)[0]].reshape(-1,)

    # evaluate the prediction
    accuracy, auc, f1 = evaluate(test_y, pred, pred_prob)

    return ab_classifier, accuracy, auc, f1

def random_forest_classification(train_x, train_y, test_x, test_y):
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(train_x, train_y)
    pred = rf_classifier.predict(test_x)
    pred_prob = rf_classifier.predict_proba(test_x)
    pred_prob = pred_prob[:,np.where(rf_classifier.classes_==1)[0]].reshape(-1,)

    # evaluate the prediction
    accuracy, auc, f1 = evaluate(test_y, pred, pred_prob)
    return rf_classifier, accuracy, auc, f1


if __name__ == "__main__":
    input_filename = "temp_data/rf_predictions_ntree128_md32_delay_response_success.csv"
    input_df = pd.read_csv(input_filename)

    input_df.dropna(subset=["response_apology","response_promise","response_ignore","response_transparency"], inplace=True)
    input_df.reset_index(inplace=True)
    print("{} delay updates".format(input_df.shape[0]))

    #################### Initialize parameters ##################
    dim_size = 60

    print('Text preprocessing...')
    texts = input_df['content']
    apology_label = input_df['response_apology'].values.astype(int)
    promise_label = input_df['response_promise'].values.astype(int)
    ignore_label = input_df['response_ignore'].values.astype(int)
    transparency_label = input_df['response_transparency'].values.astype(int)

    # preprocess text
    processed_texts = text_preprocessing(texts)
    processed_texts = processed_texts.tolist()

    stopwords = stopwords.words('english')

    # generate list of word lists in each update
    data = []
    for document in processed_texts:
        tokenized_words = word_tokenize(document)
        temp = []
        for word in tokenized_words:
            if word not in stopwords:
                temp.append(word)
        data.append(temp)


    # Create SkipGram model
    skipgram_model = Word2Vec(data, size=dim_size, iter=30, min_count=3, window=3, sg=1)

    # generate average vector representations of each update
    print('Generating SkipGram vectors...')
    skipgram_avg_vec_arr = compute_avg_vec(data, dim_size, skipgram_model)

    ## Classification
    # kf = KFold(n_splits=5)
    skf = StratifiedKFold(n_splits=5)
    ros = RandomOverSampler()

    apology_eval = {"accuracy":[], "auc":[], "f1":[]}
    promise_eval = {"accuracy":[], "auc":[], "f1":[]}
    ignore_eval = {"accuracy":[], "auc":[], "f1":[]}
    transparency_eval = {"accuracy":[], "auc":[], "f1":[]}
    for train_index, test_index in skf.split(skipgram_avg_vec_arr, apology_label):
        train_vec, test_vec = skipgram_avg_vec_arr[train_index,:], skipgram_avg_vec_arr[test_index,:]
        train_apology, test_apology = apology_label[train_index], apology_label[test_index]

        # oversampling
        # train_vec_resampled, train_apology_resampled = ros.fit_resample(train_vec, train_apology) # oversampling
        train_vec_resampled, train_apology_resampled = SMOTE().fit_resample(train_vec, train_apology) # oversampling using SMOTE
        print(train_apology_resampled.mean())

        ## classification train
        # apology
        apology_model, apology_accuracy, apology_auc, apology_f1 = adaboost_classification(train_vec_resampled, train_apology_resampled, test_vec, test_apology)
        # apology_model, apology_accuracy, apology_auc, apology_f1 = random_forest_classification(train_vec_resampled, train_apology_resampled, test_vec, test_apology)
        apology_eval["accuracy"].append(apology_accuracy)
        apology_eval["auc"].append(apology_auc)
        apology_eval["f1"].append(apology_f1)

    for train_index, test_index in skf.split(skipgram_avg_vec_arr, promise_label):
        train_vec, test_vec = skipgram_avg_vec_arr[train_index,:], skipgram_avg_vec_arr[test_index,:]
        train_promise, test_promise = promise_label[train_index], promise_label[test_index]

        # train_vec_resampled, train_promise_resampled = ros.fit_resample(train_vec, train_promise) # oversampling
        train_vec_resampled, train_promise_resampled = SMOTE().fit_resample(train_vec, train_promise) # oversampling using SMOTE

        ## classification train
        # promise
        promise_model, promise_accuracy, promise_auc, promise_f1 = adaboost_classification(train_vec_resampled, train_promise_resampled, test_vec, test_promise)
        # promise_model, promise_accuracy, promise_auc, promise_f1 = random_forest_classification(train_vec_resampled, train_promise_resampled, test_vec, test_promise)
        promise_eval["accuracy"].append(promise_accuracy)
        promise_eval["auc"].append(promise_auc)
        promise_eval["f1"].append(promise_f1)

    for train_index, test_index in skf.split(skipgram_avg_vec_arr, ignore_label):
        train_vec, test_vec = skipgram_avg_vec_arr[train_index,:], skipgram_avg_vec_arr[test_index,:]
        train_ignore, test_ignore = ignore_label[train_index], ignore_label[test_index]

        # train_vec_resampled, train_ignore_resampled = ros.fit_resample(train_vec, train_ignore) # oversampling
        train_vec_resampled, train_ignore_resampled = SMOTE().fit_resample(train_vec, train_ignore) # oversampling using SMOTE

        ## classification train
        # ignore
        ignore_model, ignore_accuracy, ignore_auc, ignore_f1 = adaboost_classification(train_vec_resampled, train_ignore_resampled, test_vec, test_ignore)
        # ignore_model, ignore_accuracy, ignore_auc, ignore_f1 = random_forest_classification(train_vec_resampled, train_ignore_resampled, test_vec, test_ignore)
        ignore_eval["accuracy"].append(ignore_accuracy)
        ignore_eval["auc"].append(ignore_auc)
        ignore_eval["f1"].append(ignore_f1)

    for train_index, test_index in skf.split(skipgram_avg_vec_arr, transparency_label):
        train_vec, test_vec = skipgram_avg_vec_arr[train_index,:], skipgram_avg_vec_arr[test_index,:]
        train_transparency, test_transparency = transparency_label[train_index], transparency_label[test_index]

        # train_vec_resampled, train_transparency_resampled = ros.fit_resample(train_vec, train_transparency) # oversampling
        train_vec_resampled, train_transparency_resampled = SMOTE().fit_resample(train_vec, train_transparency) # oversampling using SMOTE

        ## classification train
        # transparency
        transparency_model, transparency_accuracy, transparency_auc, transparency_f1 = adaboost_classification(train_vec_resampled, train_transparency_resampled, test_vec, test_transparency)
        # transparency_model, transparency_accuracy, transparency_auc, transparency_f1 = random_forest_classification(train_vec_resampled, train_transparency_resampled, test_vec, test_transparency)
        transparency_eval["accuracy"].append(transparency_accuracy)
        transparency_eval["auc"].append(transparency_auc)
        transparency_eval["f1"].append(transparency_f1)

    print("** Apology  **")
    print("Accuracy: {}, AUC: {}, F1: {}".format(mean(apology_eval['accuracy']), mean(apology_eval['auc']), mean(apology_eval['f1'])))
    print("** Promise **")
    print("Accuracy: {}, AUC: {}, F1: {}".format(mean(promise_eval['accuracy']), mean(promise_eval['auc']), mean(promise_eval['f1'])))
    print("** Ignore **")
    print("Accuracy: {}, AUC: {}, F1: {}".format(mean(ignore_eval['accuracy']), mean(ignore_eval['auc']), mean(ignore_eval['f1'])))
    print("** Transparency **")
    print("Accuracy: {}, AUC: {}, F1: {}".format(mean(transparency_eval['accuracy']), mean(transparency_eval['auc']), mean(transparency_eval['f1'])))



