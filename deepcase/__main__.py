# Imports
from sklearn.metrics import classification_report
import argformat
import argparse
import numpy  as np
import pandas as pd
import torch

# DeepCASE imports
from deepcase.preprocessing   import Preprocessor
from deepcase.context_builder import ContextBuilder
from deepcase.interpreter     import Interpreter
from deepcase.utils           import confusion_report, show_sequences

if __name__ == "__main__":

    ########################################################################
    #                           Parse arguments                            #
    ########################################################################

    # Parse arguments
    parser = argparse.ArgumentParser(
        prog            = "deepcase.py",
        description     = "DeepCASE: Semi-Supervised Contextual Analysis of Security Events",
        formatter_class = argformat.StructuredFormatter,
    )

    # Add DeepCASE mode arguments, run in different modes
    parser.add_argument('mode', help="mode in which to run DeepCASE", choices=(
        'sequence',
        'train',
        'cluster',
        'manual',
        'automatic',
    ))

    # Add I/O arguments
    group_io = parser.add_argument_group("Input/Output")
    group_io.add_argument('--csv'   , help="CSV events file to process")
    group_io.add_argument('--txt'   , help="TXT events file to process")
    group_io.add_argument('--events', default='auto', help="number of distinct events to handle")

    # Add Sequence arguments
    group_sequences = parser.add_argument_group("Sequencing")
    group_sequences.add_argument('--length'  , type=int  , default=10   , help="sequence LENGTH           ")
    group_sequences.add_argument('--timeout' , type=float, default=86400, help="sequence TIMEOUT (seconds)")
    group_sequences.add_argument('--save-sequences', help="path to save sequences")
    group_sequences.add_argument('--load-sequences', help="path to load sequences")

    # Add ContextBuilder arguments
    group_builder = parser.add_argument_group("ContextBuilder")
    group_builder.add_argument('--hidden', type=int  , default=128, help="HIDDEN layers dimension")
    group_builder.add_argument('--delta' , type=float, default=0.1, help="label smoothing DELTA")
    group_builder.add_argument('--save-builder', help="path to save ContextBuilder")
    group_builder.add_argument('--load-builder', help="path to load ContextBuilder")

    # Add Interpreter arguments
    group_interpreter = parser.add_argument_group("Interpreter")
    group_interpreter.add_argument('--confidence' , type=float, default=0.2, help="minimum required CONFIDENCE")
    group_interpreter.add_argument('--epsilon'    , type=float, default=0.1, help="DBSCAN clustering EPSILON")
    group_interpreter.add_argument('--min_samples', type=int  , default=5  , help="DBSCAN clustering MIN_SAMPLES")
    group_interpreter.add_argument('--save-interpreter', help="path to save Interpreter")
    group_interpreter.add_argument('--load-interpreter', help="path to load Interpreter")
    group_interpreter.add_argument('--save-clusters'   , help="path to CSV file to save clusters")
    group_interpreter.add_argument('--load-clusters'   , help="path to CSV file to load clusters")
    group_interpreter.add_argument('--save-prediction' , help="path to CSV file to save prediction")

    # Add Training arguments
    group_train = parser.add_argument_group("Train")
    group_train.add_argument('--epochs', type=int, default=10 , help="number of epochs to train with")
    group_train.add_argument('--batch' , type=int, default=128, help="batch size       to train with")

    # Add other arguments
    group_other = parser.add_argument_group("Other")
    group_other.add_argument('--device', default='auto'     , help="DEVICE used for computation (cpu|cuda|auto)")
    group_other.add_argument('--silent', action='store_true', help="silence mode, do not print progress")

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                     A. Security event sequences                      #
    ########################################################################

    # Create preprocessor
    preprocessor = Preprocessor(
        length  = args.length,
        timeout = args.timeout,
    )

    # Load files
    if args.csv is not None and args.txt is not None:
        # Raise an error if both csv and txt are specified
        raise ValueError("Please specify EITHER --csv OR --txt.")
    if args.csv:
        # Load csv file
        context, events, labels, mapping = preprocessor.csv(
            args.csv,
            verbose = not args.silent,
        )

    elif args.txt:
        # Load txt file
        context, events, labels, mapping = preprocessor.text(
            args.txt,
            verbose = not args.silent,
        )

    elif args.save_sequences:
        raise ValueError("Cannot save sequences without --csv or --txt input")

    elif not args.load_sequences:
        raise ValueError(
            "No input given, please provide either raw input using --csv or "
            "--txt or load preprocessed sequences using --load-sequences."
        )

    # Save sequences if necessary
    if args.save_sequences:
        with open(args.save_sequences, 'wb') as outfile:
            torch.save({
                "events" : events,
                "context": context,
                "labels" : labels,
                "mapping": mapping,
            }, outfile)

    # Load sequences if necessary
    if args.load_sequences:
        with open(args.load_sequences, 'rb') as infile:
            # Load data
            data = torch.load(infile)
            # Extract data
            events  = data["events"]
            context = data["context"]
            labels  = data["labels"]
            mapping = data["mapping"]

    # If sequence mode, output result and exit
    if args.mode == "sequence":
        mapping = None
        # Show sequences
        show_sequences(
            context  = context,
            events   = events,
            labels   = labels,
            mapping  = mapping,
            NO_EVENT = preprocessor.NO_EVENT,
        )
        exit()


    ########################################################################
    #                         Set "auto" arguments                         #
    ########################################################################

    # Automatically set device argument
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Automatically set the number of events to expect
    if args.events == "auto":
        args.events = len(mapping)
    else:
        args.events = int(args.events)

    # Cast tensors to device
    events  = events .to(args.device)
    context = context.to(args.device)

    ########################################################################
    #                          B. Context Builder                          #
    ########################################################################

    # Load the builder, if necessary
    if args.load_builder:
        context_builder = ContextBuilder.load(args.load_builder, args.device)

    # Otherwise create a new ContextBuilder
    else:
        # Create ContextBuilder
        context_builder = ContextBuilder(
            input_size    = args.events,
            output_size   = args.events,
            hidden_size   = args.hidden,
            num_layers    = 1,
            max_length    = args.length,
            bidirectional = False,
            LSTM          = False,
        ).to(args.device)

    # Training mode
    if args.mode == "train":

        # Train the ContextBuilder
        context_builder.fit(
            X             = context,
            y             = events.reshape(-1, 1),
            epochs        = args.epochs,
            batch_size    = args.batch,
            learning_rate = 0.01,
            teach_ratio   = 0.5,
            verbose       = not args.silent,
        )

    # Save the builder, if necessary
    if args.save_builder:
        context_builder.save(args.save_builder)

    ########################################################################
    #                            C. Interpreter                            #
    ########################################################################

    # Load the interpreter, if necessary
    if args.load_interpreter:
        interpreter = Interpreter.load(
            args.load_interpreter,
            context_builder = context_builder,
        )
    # Otherwise create a new Interpreter
    else:
        # Create Interpreter
        interpreter = Interpreter(
            context_builder = context_builder,
            features        = args.events,
            eps             = args.epsilon,
            min_samples     = args.min_samples,
            threshold       = args.confidence,
        )

    # Cluster samples with the interpreter
    if args.mode == "cluster":

        # Cluster samples with the interpreter
        clusters = interpreter.cluster(
            X          = context,
            y          = events.reshape(-1, 1),
            iterations = 100,
            batch_size = 1024,
            verbose    = not args.silent,
        )

        # Save clusters, if necessary
        if args.save_clusters:

            # Set labels to -1 if no labels were provided
            if labels is None:
                labels = np.full(clusters.shape[0], -1, dtype=int)

            # Save to file
            pd.DataFrame({
                'clusters': clusters,
                'labels'  : labels,
            }).to_csv(args.save_clusters, index=False)

    # Save the interpreter, if necessary
    if args.save_interpreter:
        interpreter.save(args.save_interpreter)

    ########################################################################
    #                          D. Manual analysis                          #
    ########################################################################

    if args.mode == "manual":

        # Load cluster labels from file, if necessary
        if args.load_clusters:

            # Load labels from csv file
            labels = pd.read_csv(
                args.load_clusters,
                index_col = False,
            )['labels'].values

        # Otherwise, check if labels are present in sequences
        elif labels is None:
            raise ValueError(
                "Cannot assign clusters because labels were provided. Please "
                "provide labeles either through the --load-clusters argument or"
                " by loading sequences with tagged with a label. See "
                "deepcase.preprocessor.Preprocessor for information on how to "
                "load input samples with a given label."
            )

        # Use given labels to compute score for each cluster
        scores = interpreter.score_clusters(labels, strategy="max")
        # Manually assign computed scores
        interpreter.score(scores, verbose=not args.silent)

        # Save the interpreter, if necessary
        if args.save_interpreter:
            interpreter.save(args.save_interpreter)

    ########################################################################
    #                      E. Semi-automatic analysis                      #
    ########################################################################

    if args.mode == "automatic":

        # Check whether predictions can be saved
        if args.save_prediction is None:
            raise ValueError(
                "Please use --save-prediction CSV_FILE to specify a file to "
                "save the predictions for each sequence."
            )

        # Compute predicted scores
        prediction = interpreter.predict(
            X          = context,
            y          = events.reshape(-1, 1),
            iterations = 100,
            batch_size = 1024,
            verbose    = not args.silent,
        )

        # Save to file
        pd.DataFrame({
            'labels': prediction,
        }).to_csv(args.save_prediction, index=False)


        # If labels were provided, print classification report
        if labels is not None:

            # Print classification report
            print("Classification report")
            print(classification_report(
                y_pred        = prediction,
                y_true        = labels,
                digits        = 4,
                zero_division = 0,
            ))

            # Print confusion matrix
            print("Confusion matrix")
            all_labels = np.unique(labels).tolist()
            print(confusion_report(
                y_pred       = prediction,
                y_true       = labels,
                labels       = [-3, -2, -1] + all_labels,
                target_names = ['LOW CONFIDENCE', 'NOT IN TRAIN', 'LOW EPS'] + all_labels,
                skip_x       = all_labels,
                skip_y       = ['LOW CONFIDENCE', 'NOT IN TRAIN', 'LOW EPS']
            ))
