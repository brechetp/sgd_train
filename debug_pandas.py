
import pandas as pd
import numpy as np

if __name__ == '__main__':

    NTRY=10
    # prt that have multiple tries
    #  train / test
    NEPOCH = 200
    index = pd.Index(np.arange(1, NEPOCH+1))
    names=['set', 'stat', 'layer', 'try']
    tries = np.arange(NTRY)
    sets = ['train', 'test']
    stats = ['loss', 'err']
    layers = ['last', 'hidden']
    columns=pd.MultiIndex.from_product([sets, stats, layers, tries], names=names)
    quant = pd.DataFrame(columns=columns, index=index)
    quant.sort_index(axis=1, inplace=True)
    print('lexsorted: ', quant.columns.is_lexsorted())
    print('lex sort depth: ', quant.columns.lexsort_depth)
    columns_cut = pd.MultiIndex.from_product([stats, layers, tries], names=names[1:])
#    new_entry = pd.DataFrame([(np.arange(48,58)/10).tolist(),
#                 np.arange(32, 42).tolist(),
#                np.arange(0, 10).tolist(),
#                 np.arange(1, 11).tolist()] ,
#        index=pd.Index([1]), columns=columns_cut)

    quant.loc[pd.IndexSlice[1, ('train')]] = np.arange(0, 40)
    print('{:g}'.format(quant.loc[1, ('train')].min()))
