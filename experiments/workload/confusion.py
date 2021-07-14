import numpy as np

with open('interpreter_max.clean') as infile:
    matrix = {
        'INFO'  : np.zeros(5, dtype=int),
        'LOW'   : np.zeros(5, dtype=int),
        'MEDIUM': np.zeros(5, dtype=int),
        'HIGH'  : np.zeros(5, dtype=int),
        'ATTACK': np.zeros(5, dtype=int),
    }
    mode   = None
    active = False
    for line in infile.readlines():
        # Remove trailing \n
        line = line[:-1]

        if "Mode" in line:
            mode = line.split()[1]
        elif not line:
            active = False
        elif mode == "Automatic" and "T\P" in line and 'CONF' not in line:
            active = True
        elif active:
            label, *numbers = line.split()
            numbers = np.asarray([int(n) for n in numbers])
            matrix[label] += numbers

    # Create confusion matrix
    matrix = np.stack((
        matrix['INFO'  ],
        matrix['LOW'   ],
        matrix['MEDIUM'],
        matrix['HIGH'  ],
        matrix['ATTACK'],
    ))

    print(matrix)
    print(matrix.sum())

    # Create classification report
    labels = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'ATTACK']
    a, b = np.meshgrid(labels, labels)

    from sklearn.metrics import classification_report
    print("\nClassification report - normal")
    print(classification_report(
        a.flatten(),
        b.flatten(),
        sample_weight=matrix.T.flatten(),
        digits=4,)
    )

    b[0:, 0] = 'INFO'
    b[1:, 1] = 'LOW'
    b[2:, 2] = 'MEDIUM'
    b[3:, 3] = 'HIGH'
    b[4:, 4] = 'ATTACK'
    print("\nClassification report - higher is ok")
    print(classification_report(
        a.flatten(),
        b.flatten(),
        sample_weight=matrix.T.flatten(),
        digits=4,)
    )
