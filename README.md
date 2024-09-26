# dynSBevo
The dynSBevo package provides functions for performing approximate maximum likelihood estimation of the dynamic stochastic block model for longitudinal network using two versions (standard and evolutionary) of the variational expectation maximization (VEM) algorithm. The package is implemented in C++ to provide efficient computations.

To install the `dynSBevo` package directly from GitHub:
```r
# install.packages("devtools")
require(devtools)
devtools::install_github("LB1304/dynSBevo")
```

To download the .tar.gz file (for manual installation) use [this link](https://github.com/LB1304/dynSBevo/archive/main.tar.gz).


The main functions of the package are 
- `SBdyn_draw`, to sample a random longitudinal network data from the dynamic stochastic block model; 
- `est_dynSB`, to performs approximate maximum likelihood estimation of the parameters of the dynamic stochastic block model for longitudinal network data. 

More on the options is accessible with the following commands.

```r
require(dynSBevo)

help("SBdyn_draw")
help("est_dynSB")
```
