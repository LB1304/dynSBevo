<h1 align="center">Evolutionary-based estimation of the dynamic stochastic block model</h1>
<p align="center"> <span style="font-size: 14px;"><em><strong>Luca Brusa &middot; Fulvia Pennoni</strong></em></span> </p>
<br>

The `dynSBevo` package provides functions for performing approximate maximum likelihood estimation of the dynamic stochastic block model for longitudinal network using two versions (standard and evolutionary) of the variational expectation maximization (VEM) algorithm. The package is implemented in C++ to provide efficient computations.

To install the `dynSBevo` package directly from GitHub:
```r
if (!"devtools" %in% installed.packages()) {
    install.packages("devtools", dependencies = TRUE)
}

require(devtools)
devtools::install_github("LB1304/dynSBevo")
```

To download the .tar.gz file (for manual installation) use [this link](https://github.com/LB1304/dynSBevo/archive/main.tar.gz).


The main functions of the package are: 
- `SBdyn_draw`, to sample a random longitudinal network data from the dynamic stochastic block model; 
- `est_dynSB`, to performs approximate maximum likelihood estimation of the parameters of the dynamic stochastic block model for longitudinal network data. 

More on the options of the functions and some examples are accessible with the following commands.

```r
require(dynSBevo)

help("SBdyn_draw")
help("est_dynSB")
```
