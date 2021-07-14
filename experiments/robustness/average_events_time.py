import numpy  as np
import pandas as pd
from tqdm import tqdm

from deepcase                 import DeepCASE
from deepcase.context_builder import ContextBuilder
from deepcase.interpreter     import Interpreter
from deepcase.preprocessing   import Preprocessor

if __name__ == "__main__":

    ########################################################################
    #                              Load data                               #
    ########################################################################

    path = '../../../data/preprocessed.csv'

    data           = pd.read_csv(path)[['timestamp', 'machine']][2_000_000:]
    total_time     = data['timestamp'].max() - data['timestamp'].min()
    total_machines = np.unique(data['machine'].values).shape[0]

    # # Create preprocessor
    # preprocessor = Preprocessor(
    #     context = 10,
    #     timeout = 60*60*24,
    # )
    #
    # # Load data from file
    # events, context, label, mapping = preprocessor.csv(
    #     path,
    #     verbose = True,
    # )
    # events_test  = events [2_000_000:].to('cuda').reshape(-1, 1)
    # context_test = context[2_000_000:].to('cuda')
    #
    # ########################################################################
    # #                            Load DeepCASE                             #
    # ########################################################################
    #
    # # Create DeepCASE object
    # deepcase = DeepCASE(
    #     n_features  = 280,
    #     complexity  = 128,
    #     context     = 10,
    #     device      = 'cuda',
    #     eps         = 0.1,
    #     min_samples = 5,
    #     threshold   = 0.2,
    # )
    #
    # # Load ContextBuilder
    # deepcase.context_builder = ContextBuilder.load(
    #     infile = '../baseline/context.save',
    #     device = 'cuda',
    # )
    # # Load Interpreter
    # deepcase.interpreter = Interpreter.load(
    #     infile           = '../baseline/interpreter.save',
    #     context_builder  = deepcase.context_builder,
    # )
    #
    # ########################################################################
    # #                             Predict data                             #
    # ########################################################################
    #
    # # Predict testing data
    # pred_test  = deepcase.interpreter.predict(
    #     X       = context_test ,
    #     y       = events_test ,
    #     verbose = True,
    # ).squeeze(1)
    #
    # # Transform back to labels
    # y_pred_test  = np.zeros(pred_test .shape[0])
    #
    # y_pred_test [:] = pred_test [:]
    # # Set impact - test
    # y_pred_test [pred_test  >= 0] = 0
    # y_pred_test [pred_test  >  0] = 1
    # y_pred_test [pred_test  >  1] = 2
    # y_pred_test [pred_test  >  2] = 3
    # y_pred_test [pred_test  >  3] = 4
    #
    # pred_test  = y_pred_test .astype(int)
    #
    # # Save prediction
    # np.save('prediction.npy', pred_test)
    # Load prediction
    pred_test = np.load('prediction.npy')

    ########################################################################
    #                            Compute result                            #
    ########################################################################

    # Get datapoints of alerts
    data['alert'] = pred_test < 0
    print(np.unique(data['machine'].values).shape[0])

    overall_time = 0

    frequency = list()
    for machine, frame in tqdm(data.groupby('machine')):
        timespan = frame['timestamp'].max() - frame['timestamp'].min()
        overall_time += timespan
        alerts   = frame['alert'].sum()
        alert_frequency = alerts / max(1, timespan)
        frequency.append(alert_frequency)

    frequency = np.asarray(frequency)
    print(data['alert'].sum() / overall_time)
    print()
    print(frequency.min())
    print(frequency.mean())
    print(frequency.max())
    print()
    print(frequency[frequency != 0].min())
    print(frequency[frequency != 0].mean())
    print(frequency[frequency != 0].max())
    print()
    timespan = data['timestamp'].max() - data['timestamp'].min()
    alerts   = data['alert'].sum()
    print(alerts / max(1, timespan))
    print(np.unique(data['machine'].values).shape[0])
    exit()

    alerts         = data.shape[0]
    alert_time     = data['timestamp'].max() - data['timestamp'].min()
    alert_machines = np.unique(data['machine'].values).shape[0]

    print(total_time / alerts)
    exit()
    print(total_time)
    print(alert_time)
    print(total_machines)
    print(alert_machines)
