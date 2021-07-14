with open('interpreter_max.result') as infile:
    with open('interpreter_max.clean', 'w') as outfile:
        for line in infile:
            if not line.startswith('Loading:')      and\
               not line.startswith('Predicting:')   and\
               not line.startswith('Optimizing query:')   and\
               not line.startswith('Lookup table:'):
                outfile.write(line)
