## HAZniCS-examples

Code for the numerical examples presented in "HAZniCS - Software Components for Multiphysics Problems" by A. Budisa, X. Hu, M. Kuchta, K.-A. Mardal and L. T. Zikatanov (2022).

### Requirements
- Install **HAZniCS** software: Check out this [README](https://github.com/HAZmathTeam/hazmath/blob/main/examples/haznics/README.md) file at HAZmath repo
- Additional Python packages (recommended versions): `scipy (v1.21.5)`, `sympy (v1.9)`, `tabulate (v0.8.10)`
- Download required mesh files by executing `bash downloads.sh`

### How to run demo examples
e.g. AMG for the linear elliptic problem (with HAZmath and Hypre)
- with default parameters
```
python3 demo_elliptic_test.py
```
- with specifying mesh parameters
```
python3 demo_elliptic_test.py -dim 3 -N 16 -refine 2
```
