# from adaboost import decision_stump

def adaboost_with_decision_stump(traindata, testdata):
    weights = [1.0/traindata.shape[0]]*traindata.shape[0]
    print(weights)
    max_iteration = 50000
    for i in range(max_iteration):
        decision_stump(traindata)
    return


def sampling_with_replacement(traindata, weights):
    cummulative_weights = [0.0]
    for i in range(len(weights)):
        cummulative_weights.append(cummulative_weights[i]+weights[i])

    replaced_examples = None

