import shap


def build_background_data(ds_loader, data, bg_size):
    """ Summarize a dataset with k mean samples weighted by the number of data points they
    each represent.

    Parameters
    ----------
    ds_loader : the data loader of the data set
    
    data : Matrix of data samples to summarize (# samples x # features)

    k : int
        Number of means to use for approximation.

    Returns
    -------
    DenseData object.
    """
    
    def decompose_dummified_name(col_name):
        return col_name.split("=")[0]
    
    column_key_indice = [(c if ds_loader.is_numerical(c) else decompose_dummified_name(c), idx) 
                         for idx, c in enumerate(data.columns)]

    grouping_map = dict()
    for key, indice in column_key_indice:
        grouping_map.setdefault(key, []).append(indice)
    
    group_names, groups = zip(*grouping_map.items())
    
    background_data = shap.kmeans(data, bg_size)
    background_data.groups = groups
    background_data.group_names = group_names
    background_data.groups_size = len(groups)
    
    return background_data


def make_readable_features(ds_loader, sample, group_names):
    
    raw_sample = ds_loader.data[list(group_names)].iloc[[sample.name],:]
    readable_features = [f"{f}={v}" for f, v in zip(raw_sample.columns, raw_sample.values[0])]
    
    return readable_features
    

