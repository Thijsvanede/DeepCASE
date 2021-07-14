import numpy as np
import pandas as pd
import sys
from datetime import datetime

sys.path.insert(1, '../../deepseq/')
from interpreter.utils import lookup_table
from utils             import box, header

df = pd.read_csv('../../../data/tmp2.csv')
organisations = df['source'].values

for organisation, mask in sorted(lookup_table(organisations)):
    header("Organisation: {:2}".format(organisation))
    df_ = df.loc[mask]
    print("Captured from {} to {}".format(
        datetime.fromtimestamp(df_['ts_start'].values.min()).isoformat(),
        datetime.fromtimestamp(df_['ts_start'].values.max()).isoformat(),
    ))
    print("Events  : {}".format(df_.shape[0]))
    print("Machines: {}".format(np.unique(df_['src_ip'].values).shape[0]))
    print("\nRisk levels:")
    print("\tINFO  : {}".format((df_['impact'].values[df_['breach'].values == -1] <=  0).sum()))
    print("\tLOW   : {}".format((np.logical_and(df_['impact'].values[df_['breach'].values == -1] >   0,
                                                df_['impact'].values[df_['breach'].values == -1] <  30).sum())))
    print("\tMEDIUM: {}".format((np.logical_and(df_['impact'].values[df_['breach'].values == -1] >= 30,
                                                df_['impact'].values[df_['breach'].values == -1] <  70).sum())))
    print("\tHIGH  : {}".format((df_['impact'].values[df_['breach'].values == -1] >= 70).sum()))
    print("\tATTACK: {}".format((df_['breach'].values != -1).sum()))
    print()

for organisation, mask in sorted(lookup_table(organisations)):
    df_ = df.loc[mask]

    print("{:2} & {:8} & {:8} & {:8} & {:8} & {:8} & {:8} & {:8} \\\\".format(
        organisation+1,
        np.unique(df_['src_ip'].values).shape[0],
        df_.shape[0],
        (df_['impact'].values[df_['breach'].values == -1] <=  0).sum(),
        np.logical_and(df_['impact'].values[df_['breach'].values == -1] >   0,
                       df_['impact'].values[df_['breach'].values == -1] <  30).sum(),
        np.logical_and(df_['impact'].values[df_['breach'].values == -1] >= 30,
                       df_['impact'].values[df_['breach'].values == -1] <  70).sum(),
        (df_['impact'].values[df_['breach'].values == -1] >= 70).sum(),
        (df_['breach'].values != -1).sum(),
    ))
