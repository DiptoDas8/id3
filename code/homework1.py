import pandas as pd
import id3

def convert_to_df(traindata,flag):
    rowslist = []
    for row in traindata:
        if(flag==0):
            dict = row[0]
            dict['gets_job'] = row[1]
        else:
            dict = row
        rowslist.append(dict)

    df = pd.DataFrame(rowslist)
    if(flag==0):
        cols = df.columns.tolist()
        cols = cols[1:]+cols[0:1]
        df = df[cols]
    return df

if __name__=='__main__':
    print('Hello world')
    training_data = [
    ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False),
    ({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'}, False),
    ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
    ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
    ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
    ({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, False),
    ({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, True),
    ({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
    ({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
    ({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
    ({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
    ({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, True),
    ({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'}, True),
    ({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
    ]

    traindata = convert_to_df(training_data,0)


    test_data = [
    {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "no"},
    {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "yes"},
    {"level" : "Intern"},
    {"level" : "Senior"}
    ]

    testdata = convert_to_df(test_data,1)

    id3.id3(traindata, testdata)
