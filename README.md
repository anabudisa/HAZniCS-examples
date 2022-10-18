## HAZniCS-examples

Code for the numerical examples presented in "HAZniCS - Software Components for Multiphysics Problems" by A. Budisa, X. Hu, M. Kuchta, K.-A. Mardal and L. T. Zikatanov (2022).

### Requirements
- Install **HAZniCS** software: Check out this [README](https://github.com/HAZmathTeam/hazmath/blob/0d80e757d4f1495ef4892b2628b86b4000c0ee3a/examples/haznics/README.md) file at HAZmath repo
- Download required mesh files by executing `bash downloads.sh`

### How to run demo examples
e.g. AMG for Poisson problem (with HAZmath and Hypre)
- with default parameters
```
python3 demo_poisson_test.py
```
- with specifying mesh parameters
```
python3 demo_poisson_test.py -dim 3 -N 16 -refine 2
```
