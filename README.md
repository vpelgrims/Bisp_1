***
BISP-1

Bayesian Inference of Starlight Polarization in 1D

***


``BISP-1`` implements the Bayesian method for tomographic decomposition of the plane-of-sky orientation of the magnetic field with the use of stellar polarimetry and distance developed in [Pelgrims et al., 2022, A&A, (submitted)](https://arxiv.org/abs/astro-ph/2208.02278). The method decomposes the polarization signal along distance in terms of dust polarization screens: dust clouds of the interstellar medium of our Galaxy.

``BISP-1`` has been developed in the framework of the [PASIPHAE](https://pasiphae.science) project.

Install
=======

The code can be cloned from the repository

```
git clone https://github.com/vpelgrims/bisp1.git /where/to/clone
```

and you can install the python library

```
pip install /where/to/clone
```

Use the -e option at install if you plan to make changes within ``BISP-1`` without having to reinstall at every changes.


Background
==========

The code uses the ``dynesty`` python nested sampler to analyze the polarization + distance data through a maximum-likelihood method. The polarization signal is decomposed in thin dust-polarizing layers placed along distance. Each layer is characterized by its distance (formally the parallax) as well as by its mean polarization (Stokes parameters q and u) and intrinstic scatter which accounts for turbulence.

The model and the likelihood are fully determined in [our](https://arxiv.org/abs/astro-ph/2208.02278) paper.


Usage
=====

A [jupyter notebook](https://github.com/vpelgrims/Bisp_1/blob/main/tests/Usage_Tutorial.ipynb) is provided to describe the overal structure of the code, to ease its handling, and demonstrate the several features that are included such as results display and plotting utilities. Assuming a star sample with appropriate formating, a basic usage with default setting could read as:
```
import BISP_1 as bisp
mystars = bisp.Stars(starsample) # initialization of a Stars object
mypriors = bisp.Priors(mystars)  # initialization of a Priors object
mylos = bisp.Bisp(mystars,mypriors) # initialization of the main object
# 1. run the Bayesian decomposition assuming 1 cloud along the sightline:
mylos.run_OneLayer()
# 2. run the Bayesian decomposition assuming 2 clouds along the sightline:
mylos.run_TwoLayers()
# 3. compare the performance of both tested model
mylos.printSummaryStat()
```


References
==========

If ``BISP-1`` is useful for you and your research please cite [this](https://arxiv.org/abs/astro-ph/2208.02278) PASIPHAE paper:
"Starlight-polarization-based tomography of the magnetized ISM: Pasiphae's line-of-sight inversion method"
Pelgrims, Panopoulou, Tassis, Pavlidou et al., 2022 Astronomy & Astrophysics, submitted.

---
Bisp-1 -- ReadMe
Copyright (C) 2022 V.Pelgrims
