import numpy as np
import pickle
import scipy.sparse as sp
import time
import torch
from sklearn.neighbors import KDTree

import sys
sys.path.insert(1, '../../deepseq/')
from context_builder     import ContextBuilder
from deepseq             import DeepSeq
from interpreter.cluster import Cluster2
from interpreter.utils   import lookup_table

if __name__ == "__main__":
    device = 'cuda:1'

    # Get data
    with open('../parameter/saved_data.p', 'rb') as infile:
        data = pickle.load(infile)['data']

    X = data['threat_name']['test']['X']
    y = data['threat_name']['test']['y']
    X_impact = data['impact']['test']['X']
    y_impact = data['impact']['test']['y']

    sizes = list(map(int, [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]))

    # Resulting times
    result = {
        'epoch'         : {size: list() for size in sizes},
        'query'         : {size: list() for size in sizes},
        'fingerprinting': {size: list() for size in sizes},
        'clustering'    : {size: list() for size in sizes},
        'tree'          : {size: list() for size in sizes},
        'predict'       : {size: list() for size in sizes},
    }

    for size in sizes:
        print("Size = {}".format(size))
        for k in range(10):
            print("K    = {}".format(k))
            # Select random points
            random = torch.randperm(X.shape[0])[:size]
            # Select random data
            X_ = X[random].to(device)
            y_ = y[random].to(device)
            X_impact_ = X_impact[random]
            y_impact_ = y_impact[random]

            # Create DeepSeq
            deepseq = DeepSeq(
                n_features  = 280,
                complexity  = 128,
                context     = 10,
                device      = device,
                eps         = 0.1,
                min_samples = 5,
                threshold   = 0.2,
            )

            # Time training epochs
            start = time.time()
            deepseq.context_builder.fit(X_, y_,
                batch_size = 128,
                epochs     = 10,
                verbose    = False,
            )
            result['epoch'][size].append((time.time() - start) / 10)
            print("Epoch          took: {:.4f} seconds".format(result['epoch'][size][-1]))

            # Load actual context builder
            deepseq.context_builder = ContextBuilder.load(
                infile = '../../models/context/company.save',
                device = device,
            )

            # Time query
            start = time.time()
            pred, conf, attn = deepseq.interpreter.explain(X_, y_)
            result['query'][size].append(time.time() - start)
            print("Query          took: {:.4f} seconds".format(result['query'][size][-1]))


            # Check where confidence is above threshold
            mask = (conf >= 0.2)
            start = time.time()
            fingerprints = deepseq.interpreter.fingerprint(X_[mask], attn[mask], 280)
            result['fingerprinting'][size].append(time.time() - start)
            print("Fingerprinting took: {:.4f} seconds".format(result['fingerprinting'][size][-1]))

            if fingerprints.shape[0]:

                # Wrap up
                fingerprints = np.round(fingerprints, decimals=4).toarray()

                indices_y = lookup_table(y_[mask].squeeze(1), key=lambda x: x.item(), verbose=False)
                cluster2 = Cluster2(p=1)

                start = time.time()
                for event, context_mask in indices_y:

                    # Compute clusters per event
                    clusters = cluster2.dbscan(fingerprints[context_mask],
                        eps         = 0.1,
                        min_samples = 1,
                        verbose     = False,
                    )

                result['clustering'][size].append(time.time() - start)
                print("Clustering     took: {:.4f} seconds".format(result['clustering'][size][-1]))

                tree = dict()
                start = time.time()
                for event, context_mask in indices_y:
                    fps = torch.unique(
                        torch.Tensor(fingerprints[context_mask]),
                        dim = 0,
                    )
                    tree[event] = KDTree(fps, p=1)

                result['tree'][size].append(time.time() - start)
                print("Tree           took: {:.4f} seconds".format(result['tree'][size][-1]))

                start = time.time()
                for event, context_mask in indices_y:
                    fps = torch.unique(
                        torch.Tensor(fingerprints[context_mask]),
                        dim = 0,
                    )
                    distance, neighbours = tree[event].query(fps,
                        k               = 1,
                        return_distance = True,
                        dualtree        = fps.shape[0] >= 1e3,
                    )

                result['predict'][size].append(time.time() - start)
                print("Predict        took: {:.4f} seconds".format(result['predict'][size][-1]))


    for operation, sizes in result.items():
        print("Operation: {}".format(operation))
        for size, times in sizes.items():
            print("\tSize = {:8} -- {:.4f} seconds".format(size, np.asarray(times).mean()))
