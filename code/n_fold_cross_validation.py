from math import floor
import pandas as pd
import id3
import adaboost


def generate_n_fold_data(data, n):
    rowCount = data.shape[0]
    foldSize = floor(rowCount / n)
    for i in range(n):
        print('\n Running ' + str(i) + ' th fold cross validation')
        emdf = pd.DataFrame()
        tstd = data[i * foldSize:i * foldSize + foldSize]
        testdata = pd.concat([emdf, tstd], ignore_index=True)
        if i == 0:
            td1 = pd.DataFrame()
        else:
            td1 = data[0:i * foldSize]
        td2 = data[i * foldSize + foldSize:rowCount]
        traindata = pd.concat([td1, td2], ignore_index=True)
        # print('train: ', traindata.shape[0], 'test: ', testdata.shape[0])
        predictions = id3.id3(traindata, testdata)
        # predictions = adaboost.adaboost_with_decision_stump(traindata, testdata)
        actualclasses = testdata.iloc[:, -1].tolist()
        confusionmatrix(actualclasses, predictions)


def confusionmatrix(actualclasses, predictions):
    actualclasses = [str(x) for x in actualclasses]
    predictions = [str(x) for x in predictions]

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(actualclasses)):
        if actualclasses[i] == predictions[i] and predictions[i] == '1':
            true_positive += 1
        elif actualclasses[i] == predictions[i] and predictions[i] == '0':
            true_negative += 1
        elif actualclasses[i] != predictions[i] and predictions[i] == '1':
            false_positive += 1
        elif actualclasses[i] != predictions[i] and predictions[i] == '0':
            false_negative += 1

    # print(true_positive, ' ', true_negative, ' ', false_positive, ' ', false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    f1score = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    print('precision: ', precision, '\nrecall: ', recall, '\naccuracy: ', accuracy, '\nf1 score: ', f1score)

    return
