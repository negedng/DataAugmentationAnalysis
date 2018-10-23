import numpy as np
from sklearn.utils import check_random_state

def lists_from_dict(data_dictionary, shuffle=True):
    """Generates same length X,y 
    from a dictionary where data_dict[y[key]] is 
    the X array of the 'key' labelled data
    Parameters
    ----------
    data_dictionary : dictionary
        A dictionary: Keys - labels, values - data
    shuffle : boolean, optional
        Shuffle the return data. Default value: true.
    Returns
    -------
    array_like
        The data in an array
    array_like
        The labels (keys in the dictionary)"""
    X = []
    y = []
    for key in data_dictionary.keys():
        X = X + data_dictionary[key]
        y = y + [key]*len(data_dictionary[key])
    X = np.array(X)
    y = np.array(y)
    if(shuffle):
        random_state = check_random_state(0)
        perm = random_state.permutation(len(y))
        return X[perm], y[perm]
    return X, y

def reduce_class_samples(data, label_key=None,
                        proportion=0.2):
    """Reduce the samples in a label to the given value of it
    Parameters
    ----------
    data : dictionary
        The data dictionary, each element is array-like
    label_key : key
        The key of the class to be reduced in the array
        Default: None (choose the first in the key list)
    proportion: double
        The 0..1 value of the remaining size of the array
        Default: 0.2"""
    if(label_key==None):
        label_key = data.keys()[0]
    label_size = len(data[label_key])
    data_port = (int) (label_size*proportion)
    data[label_key] = data[label_key][:data_port]
    return data

def generate_balanced_dictionary(X,y,
                                 label_number=None):
    """Returns with a dictionary with same sample size
    Parameters
    ----------
    X : array_like
        The training data.
    y : array_like
        the labels
    label_number : int
        The number of labels to be kept.
    Returns
    -------
    data_dict
        Dictionary of arrays: key - label,
        value - data"""
    data_dict = {}
    unique, counts = np.unique(y, return_counts=True)
    if(label_number==None):
        label_number = len(unique)
    y_sort = [y_sort for _,y_sort in sorted(zip(counts,unique))]
    y_keep = y_sort[-label_number:]
    size = counts[unique.tolist().index(y_keep[0])]
    
    for i in range(len(X)):
        if y[i] in y_keep:
            if data_dict.has_key(y[i]):
                if(len(data_dict[y[i]])<size):
                    data_dict[y[i]].append(X[i])
            else:
                data_dict[y[i]] = [X[i]]
    return data_dict

