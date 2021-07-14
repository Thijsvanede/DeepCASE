import argformat
import argparse
import numpy  as np
import pandas as pd
import pickle
import sys
import torch
import tqdm

sys.path.insert(1, '../../deepseq/')
from preprocessing import NONE


if __name__ == "__main__":
    ########################################################################
    #                           Parse arguments                            #
    ########################################################################
    # Parse arguments
    parser = argparse.ArgumentParser(
        description     = "Handle pretrained data",
        formatter_class = argformat.StructuredFormatter,
    )

    # Add arguments
    parser.add_argument('file', help='file to parse')
    parser.add_argument('--dim-input', type=int  , default=10   , help="length of input sequence")
    parser.add_argument('--time'     , type=float, default=86400, help="max time length of input sequence")
    parser.add_argument('--pickle'   , help="pickle input file to read from")
    parser.add_argument('--spickle'  , help="pickle output file to write to")

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                             Process data                             #
    ########################################################################

    if args.pickle is not None:
        with open(args.pickle, 'rb') as infile:
            result = pickle.load(infile)

    else:
        # Read data
        df = pd.read_csv(args.file)

        # Initialise result
        result = dict()

        # Group by organisation
        for organisation, data in df.groupby('source'):
            # Initialise result for organisation
            result[organisation] = {
                'threat_name': list(),
                'impact'     : list(),
                'X'          : list(),
                'y'          : list(),
            }

            # Sort by values
            data = data.sort_values('ts_start')

            NONE = int(NONE)

            # Group by machine
            for machine, events in tqdm.tqdm(data.groupby('src_ip')):

                # Loop over all sequences
                for sequence in events.rolling(args.dim_input + 1):
                    # Collect threats
                    threats = sequence['threat_name'].values
                    # Collect timestamps
                    times   = sequence['ts_start'].values
                    # Remove values earlier than args.time seconds
                    threats2 = times[times >= times[-1] - args.time]

                    # Pad with NONE if necessary
                    if threats.shape[0] != args.dim_input+1:
                        threats = np.concatenate((
                            [NONE]*(args.dim_input+1-threats.shape[0]),
                            threats,
                        ))

                    # Collect breach and impact
                    breach  = sequence['breach'].values[-1]
                    impact  = sequence['impact'].values[-1]

                    # Get input and output
                    X = threats[:-1]
                    y = threats[-1:]

                    # Set values
                    result[organisation]['threat_name'].append(threats)
                    result[organisation]['impact'     ].append([impact, breach])
                    result[organisation]['X'          ].append(X)
                    result[organisation]['y'          ].append(y)

            # Transform to Tensors
            for k, v in result[organisation].items():
                result[organisation][k] = torch.as_tensor(v)

        with open(args.spickle, 'wb') as outfile:
            pickle.dump(result, outfile)

    ########################################################################
    #                             Process data                             #
    ########################################################################
