from collections     import Counter
from datetime        import datetime, timedelta
from sklearn.metrics import classification_report, homogeneity_score
import argformat
import argparse
import logging
import numpy as np
import pickle
import sys
import torch
import tqdm
sys.path.insert(1, '../../deepseq/')

from deepseq           import DeepSeq
from context_builder   import ContextBuilder
from interpreter       import Interpreter
from preprocessing     import PreprocessLoader
from utils             import box, confusion_report, header
from interpreter.utils import lookup_table

if __name__ == "__main__":
    # Set logging
    logging.basicConfig(level=logging.DEBUG, filename='logging.log')

    ########################################################################
    #                           Parse arguments                            #
    ########################################################################
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog            = "deepseq.py",
        description     = "DeepSeq: providing meta-level contextual analysis of security alerts",
        formatter_class = argformat.StructuredFormatter,
    )

    # Input
    group_input = parser.add_argument_group("Input")
    group_input.add_argument('file'      ,                            help="read preprocessed input     file")
    group_input.add_argument('test'      , nargs='?' ,                help="read preprocessed test      file")
    group_input.add_argument('malicious' , nargs='?' ,                help="read preprocessed malicious file")
    group_input.add_argument('--offset'  , type=float, default=0    , help="offset for items to load")
    group_input.add_argument('--time'    , type=float, default=86400, help="max time length of input sequence")
    group_input.add_argument('--all'     , dest='all'     , action='store_true', help="perform experiment on all data")
    group_input.add_argument('--breach'  , dest='breach'  , action='store_true', help="read breaches")
    group_input.add_argument('--incident', dest='incident', action='store_true', help="read incidents")
    group_input.add_argument('--ignore'  , dest='ignore'  , action='store_true', help="ignore incident and breach info")
    group_input.add_argument('-p', '--pickle', help="treat PICKLE as pickled object")
    group_input.add_argument('--psave', help="save data as pickled object")
    group_input.add_argument('--timedelta', type=int, default=30, help="number of days per update")

    # ContextBuilder parameters
    context_builder = parser.add_argument_group("ContextBuilder")
    context_builder.add_argument('-f', '--features'      , type=int  , default=280,          help="maximum number of expected features")
    context_builder.add_argument('-i', '--dim-input'     , type=int  , default=10,           help="length of input sequence")
    context_builder.add_argument('-o', '--dim-output'    , type=int  , default=1,            help="length of output sequence")
    context_builder.add_argument('-m', '--max-sequences' , type=float, default=float('inf'), help="maximum number of sequences ro read from input")
    context_builder.add_argument('-n', '--max-events'    , type=float, default=float('inf'), help="maximum number of events to read from input")
    context_builder.add_argument('-c', '--complexity'    , type=int  , default=128,          help="complexity of the model")

    # Training
    group_training = parser.add_argument_group("ContextBuilder training")
    group_training.add_argument('-b', '--batch-size', type=int, default=128   , help="batch size")
    group_training.add_argument('-d', '--device'    , type=str, default='auto', help="train using given device (cpu|cuda|auto)")
    group_training.add_argument('-e', '--epochs'    , type=int, default=10    , help="number of epochs to train with")
    group_training.add_argument('-r', '--random'    , action='store_true'     , help="train with random selection")
    group_training.add_argument('-s', '--silent', dest='verbose', action='store_false', help="supress printing progress")
    group_training.add_argument('--train', type=float, default=0.5, help="training samples to use (or ratio of if 0 <= TRAIN <= 1)")

    # Interpreter parameters
    interpreter = parser.add_argument_group("Interpreter")
    interpreter.add_argument('--epsilon'    , type=float, default=0.1, help="interpreter epsilon     for clustering")
    interpreter.add_argument('--min-samples', type=int,   default=5,   help="interpreter min_samples for clustering")
    interpreter.add_argument('--threshold'  , type=float, default=0.2, help="interpreter confidence threshold for fingerprinting")

    # Store/load model
    group_model = parser.add_argument_group("Model I/O parameters")
    group_model.add_argument('--load-context'    , help="load context builder from LOAD file")
    group_model.add_argument('--load-interpreter', help="load interpreter     from LOAD file")
    group_model.add_argument('--save-context'    , help="save context builder to   SAVE file")
    group_model.add_argument('--save-interpreter', help="save interpreter     to   SAVE file")

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                              Set device                              #
    ########################################################################

    # Set device if necessary
    if args.device is None or args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ########################################################################
    #                            Create DeepSeq                            #
    ########################################################################

    # Initialise deepseq
    deepseq = DeepSeq(
        n_features  = args.features,
        complexity  = args.complexity,
        context     = args.dim_input,
        device      = args.device,
        eps         = args.epsilon,
        min_samples = args.min_samples,
        threshold   = args.threshold,
    )

    ########################################################################
    #                            Load original                             #
    ########################################################################

    deepseq.context_builder = ContextBuilder.load(
        infile = '../../models/context/company.save',
        device = args.device,
    )

    deepseq.interpreter = Interpreter.load(
        infile          = '../challenges/interpreter_max.save',
        context_builder = deepseq.context_builder,
    )

    ########################################################################
    #                               Analysis                               #
    ########################################################################

    for label, models in sorted(deepseq.interpreter.labels.items()):
        scores = np.vstack([v for k, v in sorted(models.items())])
        attack = (scores[:, 1] > 0).sum()
        scores2 = scores[scores[:, 1] <= 0][:, 0]
        info   = (scores2 <   5).sum()
        low    = np.logical_and(scores2 >=  5, scores2 < 30).sum()
        medium = np.logical_and(scores2 >= 30, scores2 < 70).sum()
        high   = (scores2 >= 70).sum()
        assert info+low+medium+high+attack == scores.shape[0]
        print("{:3}: {:8} {:8} {:8} {:8} {:8}".format(
            label, info, low, medium, high, attack,
        ))

    print(len(deepseq.interpreter.labels))
