# Imports
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse arguments for given program."""
    # Create parser
    parser = argparse.ArgumentParser(
        prog = "txt2csv",
        description = "Convert a txt-compatible file to a csv-compatible file",
    )

    # Add argument
    parser.add_argument('txt', help='path to input txt file')
    parser.add_argument('csv', help='path to output csv file')
    
    # Parse arguments and return
    return parser.parse_args()


def main() -> None:
    """Main method executed when program is run."""
    # Parse arguments
    args = parse_args()
    
    # Read data
    with open(args.txt) as infile:
        data = infile.read()

    # Initialise data
    machine = []
    event = []

    # Split per machine
    for index, machine_ in tqdm(enumerate(data.split('\n'))):
        # Split into sequences
        sequence = list(map(int, machine_.split()))
        # Create csv
        machine.append(np.full(len(sequence), index))
        event.append(sequence)

    event = np.concatenate(event)
    machine = np.concatenate(machine)

    df = pd.DataFrame({
        'timestamp': np.arange(event.shape[0]),
        'machine': machine,
        'event': event,
    })

    df.to_csv(args.csv, index=None)


if __name__ == "__main__":
    main()
