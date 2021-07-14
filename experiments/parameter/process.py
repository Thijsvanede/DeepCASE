with open('optimize_input_unbuffered.result') as infile:
    for line in infile:
        if  line.startswith('[Epoch') or\
            line.startswith('Predicting:') or\
            line.startswith('Lookup table:') or\
            line.startswith('Optimizing query:') or\
            line.startswith('Fitting interpreter:') or\
            line.startswith('Clustering:'):
            continue

        elif line.startswith('Size'):
            size = int(line.split('=')[1].strip())

        elif line.startswith('Time'):
            time = int(line.split('=')[1].strip())

        elif line.startswith("Anomalous"):
            print(size, time, line.split()[1].split('/'))
