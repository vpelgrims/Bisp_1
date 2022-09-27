***
BISP-1

Bayesian Inference of Starlight Polarization in 1D

***


``BISP-1`` implements the Bayesian method for tomographic decomposition of the plane-of-sky orientation of the magnetic field with the use of stellar polarimetry and distance developed in `Pelgrims et al., 2022, A&A, (submitted) <https://arxiv.org/abs/astro-ph/2208.02278>` The method decomposes the polarization signal along distance in terms of dust polarization screens: dust clouds of the interstellar medium of our Galaxy.

``BISP-1`` has been developed in the framework of the [PASIPHAE](https://pasiphae.science) project.

.. .. contents:: **Table of Contents** */

Install
=======

The code can be cloned from the repository

.. code: shell

 git clone https://github.com/vpelgrims/bisp1.git /where/to/clone

and you can install the python library

.. code:: shell

 pip install /where/to/clone
 
Use the -e option at install if you plan to make changes within ``BISP-1`` without having to reinstall at every changes.


Background
==========

The code uses the ``dynesty`` python nested sampler to analyze the polarization + distance data through a maximum-likelihood method. The polarization signal is decomposed in thin dust-polarizing layers placed along distance. Each layer is characterized by its distance (formally the parallax) as well as by its mean polarization (Stokes parameters q and u) and intrinstic scatter which accounts for turbulence.

The model and the likelihood are fully determined in `Pelgrims V. et al., 2022, A&A, submitted.`


Usage
=====

A jupyter notebook (pick the ref.) is provided to ease the handling of the code and demonstrate the several features that are included such as results display and plotting utilities.

References
==========

If ``BISP-1`` is useful for you and your research please cite [this](https://arxiv.org/abs/astro-ph/2208.02278) PASIPHAE paper:
"Starlight-polarization-based tomography of the magnetized ISM: Pasiphae's line-of-sight inversion method"
Pelgrims, Panopoulou, Tassis, Pavlidou et al., 2022 Astronomy & Astrophysics, submitted.

---
bisp-1 -- ReadMe
Copyright (C) 2022 V.Pelgrims
# Bisp_1
