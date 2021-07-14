with open('optimize_input_unbuffered_100.result') as infile:
    for line in infile:

        if  line.startswith('[Epoch') or\
            line.startswith('Fitting interpreter') or\
            line.startswith('Lookup table') or\
            line.startswith('Clustering') or\
            line.startswith('Predicting') or\
            line.startswith('Optimizing query'):
            continue

        if line.strip():

            if line.startswith('Size'):
                size = int(line.split('=')[1].strip())
            elif line.startswith('Time'):
                time = int(line.split('=')[1].strip())
            elif line.startswith('Anomalous'):
                anomalous, total = map(int, line.split()[1].split('/'))
                print(anomalous, total)
            # else:
            #     print(line, end='')
