import numpy as np


class Tree(object):
    attribute = None
    label = None
    value = None
    children = []

    def __init__(self, attribute, label, value):
        self.attribute = attribute
        self.label = label
        self.value = value


predictions = []
threshold_depth = 50
current_depth = 0
model = None


def id3(traindata, testdata):
    # print('inside id3\ntrain: ', traindata.shape[0], 'test: ', testdata.shape[0])
    global predictions
    predictions = []
    current_depth = 0
    global model
    model = train(traindata)
    print(model)
    global most_common_class
    most_common_class = str(traindata.iloc[:, -1].value_counts().idxmax())
    test(model, testdata)
    print('Predictions: ', predictions)
    return predictions
    # path = drawTree(root)


def generate_id3(traindata):
    root = None
    node_ops = [None,
                None]  # first index is attribute and the second index is the label or a dictionary of possible branches

    '''# Check if the tree reached the threshold depth
    global current_depth
    # print('tree goes to depth '+str(current_depth))
    if current_depth >= threshold_depth:
        root = Tree(None, traindata.iloc[:, -1].value_counts().idxmax(), None)
        label = traindata.iloc[:, -1].value_counts().idxmax()
        # print(label)
        # model.append(('label', traindata.iloc[:, -1].value_counts().idxmax()))
        return root, label

    current_depth += 1'''

    # Check if all the class attributes are same
    if len(traindata.iloc[:, -1].unique()) == 1:
        root = Tree(None, traindata.iat[0, -1], None)
        label = traindata.iat[0, -1]
        # print(label)
        # model.append(('label', traindata.iat[0, -1]))
        return root, label

    # Check if the attribute list exhausted
    if len(traindata.columns) == 0:
        # print('attribute list exhausted')
        root = Tree(None, traindata.iloc[:, -1].value_counts().idxmax(), None)
        label = traindata.iloc[:, -1].value_counts().idxmax()
        # print(label)
        # model.append(('label', traindata.iloc[:, -1].value_counts().idxmax()))
        return root, label

    igs = {}
    for attr in list(traindata.columns.values):
        if attr != list(traindata.columns.values)[-1]:
            igs[attr] = infoGain(traindata, attr)

    test_attr = max(igs.keys(), key=(lambda k: igs[k]))
    # print('CHOOSING: ' + test_attr)
    # model.append(('attribute', test_attr))
    node_ops[0] = test_attr
    root = Tree(test_attr, None, None)

    attr_vals = np.unique(traindata[test_attr])
    i = 0
    branch_dict = dict()
    branch_dict['None'] = traindata.iloc[:, -1].value_counts().idxmax()
    for atr in attr_vals:
        # print('FOR VALUE: ' + str(atr))
        # model.append(('value', atr))
        root.children.append(Tree(None, None, atr))
        branch = root.children[i]
        i += 1
        subtraindata = traindata.loc[traindata[test_attr] == atr]

        if subtraindata.empty:
            branch.children.append(Tree(None, traindata.iloc[:, -1].value_counts().idxmax(), None))
        else:
            subtraindata = subtraindata.drop(test_attr, axis=1)
            # traindata.drop(test_attr, axis=1)
            tempnode, label = generate_id3(subtraindata)
            branch.children.append(tempnode)
            branch_dict[str(atr)] = label

    node_ops[1] = branch_dict

    return root, node_ops


def entropy(entClmnData):
    ent = 0.0
    val, counts = np.unique(entClmnData, return_counts=True)
    probs = counts.astype('float') / len(entClmnData)
    for p in probs:
        if p != 0.0:
            ent -= p * np.log2(p)
    return ent


def infoGain(traindata, attr):
    ig = entropy(traindata.iloc[:, -1])
    tot_cnt = traindata.shape[0]
    data_vals, data_freqs = np.unique(traindata[attr], return_counts=True)
    attr_vals_freq = dict(zip(data_vals, data_freqs))

    for attr_val, attr_freq in attr_vals_freq.items():
        ig -= attr_freq / tot_cnt * entropy((traindata.loc[traindata[attr] == attr_val]).iloc[:, -1])

    return ig


def train(traindata):
    tree_root, model = generate_id3(traindata)
    return model


def test(model, testdata):
    # print('hi from test'+ str(most_common_class))
    for i in range(testdata.shape[0]):
        sample = testdata[i:i + 1]
        sample = sample.iloc[0]
        classify(sample, model)


def classify(sample, model):
    if isinstance(model, (list,)):
        col_to_check = str(model[0])
        col_correspoding_val_in_sample = str(sample[col_to_check])
        dict_of_branches = model[1]
        branch_names_str_list = []
        for branch_name in dict_of_branches.keys():
            branch_names_str_list.append(str(branch_name))
            if str(branch_name) == col_correspoding_val_in_sample:
                classify(sample, dict_of_branches[str(branch_name)])
        # Do step for the unexpected value of branch
        if col_correspoding_val_in_sample not in branch_names_str_list:
            predictions.append(dict_of_branches['None'])
            # return dict_of_branches['None']
    else:
        predictions.append(model)
        return model
