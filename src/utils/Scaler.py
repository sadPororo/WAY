import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



def fit_scaler(corpus, feature_index, scaler_type='standard'):
    INSTANCE_INDEX, SCALER_TARGETS = feature_index
    def _generator(corpus):
        for instance in tqdm(corpus):
            yield instance[INSTANCE_INDEX.index('subsequence_stack')][:-1, SCALER_TARGETS]
            # subsequence_stack : (sum(subseq_length)+1, n_features)

    print('   ::fitting scaler for subsequential features...')
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()

    scaler.fit(np.vstack([subseq for subseq in _generator(corpus)]))
    
    return scaler

