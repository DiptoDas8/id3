import n_fold_cross_validation
import pandas as pd

if __name__=='__main__':
    print('hello world')
    data = pd.read_csv('../data/cancer_data_set-no_missing_data.csv')
    n_fold_cross_validation.generate_n_fold_data(data, 10)
