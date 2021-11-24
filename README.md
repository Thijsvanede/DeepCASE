# DeepCASE
This repository contains the code for DeepCASE by the authors of the IEEE S&P DeepCASE [1] paper [[PDF]](https://vm-thijs.ewi.utwente.nl/static/homepage/papers/deepcase.pdf).
Please [cite](#References) DeepCASE when using it in academic publications. This `main` branch provides DeepCASE as an out of the box tool. For the original experiments from the paper, please checkout the `sp` branch.

## Introduction
DeepCASE introduces a semi-supervised approach for the contextual analysis of security events. This approach automatically finds correlations in sequences of security events and clusters these correlated sequences. The clusters of correlated sequences are then shown to security operators who can set policies for each sequence. Such policies can ignore sequences of unimportant events, pass sequences to a human operator for further inspection, or (in the future) automatically trigger response mechanisms. The main contribution of this work is to reduce the number of manual inspection security operators have to perform on the vast amounts of security events that they receive.

## Documentation
We provide an extensive documentation including installation instructions and reference at [deepcase.readthedocs.io](https://deepcase.readthedocs.io/en/latest/).

## Dataset
This research uses two datasets for its evaluation:
 1. Lastline dataset.
 2. HDFS dataset.

---
**NOTE**

**The Lastline dataset was obtained under an NDA and therefore, unfortunately, we cannot share the dataset.**

---

For more details, please see the [`data`](/data) directory of this repository.

## References
[1] `van Ede, T., Aghakhani, H., Spahn, N., Bortolameotti, R., Cova, M., Continella, A., van Steen, M., Peter, A., Kruegel, C. & Vigna, G. (2022, May). DeepCASE: Semi-Supervised Contextual Analysis of Security Events. In 2022 Proceedings of the IEEE Symposium on Security and Privacy (S&P). IEEE.`

### Bibtex
```
@inproceedings{vanede2020deepcase,
  title={{DeepCASE: Semi-Supervised Contextual Analysis of Security Events}},
  author={van Ede, Thijs and Aghakhani, Hojjat and Spahn, Noah and Bortolameotti, Riccardo and Cova, Marco and Continella, Andrea and van Steen, Maarten and Peter, Andreas and Kruegel, Christopher and Vigna, Giovanni},
  booktitle={Proceedings of the IEEE Symposium on Security and Privacy (S&P)},
  year={2022},
  organization={IEEE}
}
```
