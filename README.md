# ToMExO
Tree of Mutually Exclusive Oncogenes

## MCMC moves demo

The notebook demo.ipynb provides a comprehenisve illustration of various structural moves using an example.

## Synthetic data experimments

The experiments can be reproduced using syn_datagen.py and syn_analysis.py

## Biological data analysis

### Data

The data can be downloaded from [GDAC Firehose](https://gdac.broadinstitute.org). We use Mutation_Packager_Calls (MD5) files as the input in our preprocessing steps. The IntOGen lists of potentially driver genes can be downloaded from [intogen website](https://www.intogen.org/). For Glioblastoma multiforme for instance, the table including the list of driver genes can be downloaded from [here](https://www.intogen.org/search?cohort=TCGA_WXS_GBM).

### Perprocessing

Use gdac_preproc.py to prepare the input csv file for the algorithm.

### How to run
```
$ python tomexo.py -i input.csv -o output --n_chains 10 --n_mixing 5 --n_samples 100
```

* n_chains default is 10
* n_mixing default is 0
* n_samples default is 100000

### Perprocessing

The script in gdac_postproc.py performs a basic postprocessing and generates a pdf file of the final progression model.
