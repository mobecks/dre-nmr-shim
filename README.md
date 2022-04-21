#Deep regression with ensembles enables fast, first-order shimming in low-field NMR

This is the official project page including the [paper](https://doi.org/10.1016/j.jmr.2022.107151), code, models and a link to the dataset.

## Description

Shimming in the context of nuclear magnetic resonance aims to achieve a uniform magnetic field distribution, as perfect as possible, and is crucial for useful spectroscopy and imaging. Currently, shimming precedes most acquisition procedures in the laboratory, and this mostly semi-automatic procedure often needs to be repeated, which can be cumbersome and time-consuming. The paper investigates the feasibility of completely automating and accelerating the shimming procedure by applying deep learning (DL). We show that DL can relate measured spectral shape to shim current specifications and thus rapidly predict three shim currents simultaneously, given only four input spectra. Due to the lack of accessible data for developing shimming algorithms, we also introduce a database that served as our DL training set, and allows inference of changes to 1H NMR signals depending on shim offsets. In situ experiments of deep regression with ensembles demonstrate a high success rate in spectral quality improvement for random shim distortions over different neural architectures and chemical substances. This paper presents a proof-of-concept that machine learning can simplify and accelerate the shimming problem, either as a stand-alone method, or in combination with traditional shimming methods. Our database and code are publicly available.

### Paper

Paper published at Journal of Magnetic Resonance: [https://doi.org/10.1016/j.jmr.2022.107151](https://doi.org/10.1016/j.jmr.2022.107151).

M.Becker, M.Jouda, A.Kolchinskaya, J.G.Korvink, Deep regression with ensembles enables fast, first-order shimming in low-field NMR, Journal of Magnetic Resonance 2022, 107151, ISSN 1090-7807, https://doi.org/10.1016/j.jmr.2022.107151. (https://www.sciencedirect.com/science/article/pii/S109078072200009X)


### ShimDB 

For more information and downloading of ShimDB, see [this page](https://github.com/mobecks/ShimDB).

## Execution

Almost all scripts have a dictionary variable ```initial_config``` within their first lines, which allows for easy modifications, e.g. to meta type, architecture choices and hyperparameters.

### Weak learners

First, use ```Prepickle_Data.py``` to pre-load the ShimDB database into numpy-array structure for faster loading times.
Then, run ```$python DeepRegression.py --raytuning 1``` to start the limited neural architecture search (NAS) of level-0 models.
Via ```$python DeepRegression.py```, a single model can be trained.


### Ensembles (meta-models)

Requirements: Raytune results.

The command ```python Ensembles.py --raytuning 1``` triggers raytuning of ensembles with different meta models.
With ```python Ensembles.py --raytuning 0```, the final models can be trained with the hyperparameters as specified in ```initial_config```.

### In-situ experiments

Requirements: Raytune results and trained meta models.

Experiments are conducted on a Spinsolve 80 spectrometer (Magritek GmbH, Aachen, Germany, www.magritek.com) with an interface to the Spinsolve Expert software. Scripts that need to be executed on the machine are located in the folder "MagritecScripts/".

Use ```RUN_ALL.py``` to run all DRE experiments for water sample including comparison to the downhill simplex method. 
For selective experiments use ```DRE_experiments.py```.

**Note**: Due to licence restrictions, specific files for communication to the spectrometer and triggering the experiments may be missing.


## Dependencies
Code was developed and tested with the following packages: 

- conda v4.10.10 (including all default packages) [Installation guide](https://docs.anaconda.com/anaconda/install/index.html)
- python v3.8.11
- pytorch v1.8.1 [Installation guide](https://pytorch.org/get-started/locally/)
- nmrglue v0.9.dev0 [Github](https://github.com/jjhelmus/nmrglue)
- ray v1.4.1 (+ ray tune) [Website](https://docs.ray.io/en/latest/tune/index.html)
- scienceplots v1.0.9 [Github](https://github.com/garrettj403/SciencePlots)


## Citation

If you find this method useful and want to cite it, please use the following bibtex entry:

```
@article{becker2022,
title = {Deep regression with ensembles enables fast, first-order shimming in low-field NMR},
journal = {Journal of Magnetic Resonance},
volume = {336},
pages = {107151},
year = {2022},
issn = {1090-7807},
doi = {https://doi.org/10.1016/j.jmr.2022.107151},
author = {Moritz Becker and Mazin Jouda and Anastasiya Kolchinskaya and Jan G. Korvink},
}
```