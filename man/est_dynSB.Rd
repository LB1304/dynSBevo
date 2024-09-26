\name{est_dynSB}
\alias{est_dynSB}
\title{Estimation of the dynamic stochastic block model for longitudinal network data}
\description{
This function performs approximate maximum likelihood estimation of the parameters of the dynamic stochastic block model for longitudinal network data. Two different versions of the variational expectation-maximization algorithm are implemented: standard (VEM) and evolutionary (EVEM).
}
\usage{
est_dynSB(data, k, start = NULL, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, 
          StartVal = NULL, algorithm = c("VEM", "EVEM"), 
          evolutionaryOptions = list(n_parents = NULL, LMP = NULL, UMP = NULL, R = NULL))
}
\arguments{
  \item{data}{The adjacency \code{array} of the longitudinal network (n x n x T)}
  \item{k}{An integer specifying the number of latent blocks}
  \item{start}{Initialization strategy for the VEM algorithm: \code{1} for random sampling, \code{2} for k-means algorithm, \code{3} for k-medoids clustering, \code{4} for global spectral clustering, \code{5} for static stochastic block model on the binarized aggregated network, \code{6} for weighted static stochastic block model on the aggregated network, and \code{7} for manual input of the parameters}
  \item{tol_lk, tol_theta}{Tolerance levels for convergence of the (E)VEM algorithm}
  \item{maxit}{Maximum number of iterations for the (E)VEM algorithm}
  \item{StartVal}{Starting values for the VEM algorithm; default is \code{NULL}}
  \item{algorithm}{Version of the VEM algorithm employed: \code{"VEM"} for the standard VEM and \code{"EVEM"} for the evolutionary VEM algorithms}
  \item{evolutionaryOptions}{A list containing the options for the evolutionary VEM algorithm (see the \emph{Details} section)}
}
\details{
The list \code{evolutionaryOptions} contains the evolutionary constants used to set the EVEM algorithm optimally:
\itemize{
  \item \code{n_parents} number of parents;
  \item \code{LMP, UMP} lower and upper bounds, respectively, of the interval from which the mutation probability is sampled, following a uniform distribution U(a, b);
  \item \code{R} number of cycles of the standard VEM algorithm in each update step.
}
}
\value{
  \item{LogLik}{Maximum log-likelihood function at convergence of the (E)VEM algorithm}
  \item{J}{Maximum ELBO function at convergence of the (E)VEM algorithm}
  \item{J_vec}{ELBO trace at every step of the (E)VEM algorithm}
  \item{it}{Number of iterations of the (E)EM algorithm}
  \item{piv}{Estimate of the initial probability vector (k)}
  \item{Pi}{Estimate of the transition probability matrix (k x k)}
  \item{B}{Estimate of the connection probability matrix (k x k)}
  \item{k}{Number of latent blocks}
  \item{V}{Matrix of the posterior probabilities for each unit and latent block (n x k)}
  \item{Tau1}{Variational parameters approximating the conditional probability \eqn{p(Z_i^{(1)} = u | \mathbf{Y})} (n x k)}
  \item{TAU}{Variational parameters approximating the conditional probability \eqn{p(Z_i^{(t)} = u | Z_i^{(t-1)} = v, \mathbf{Y})} (n x k x k x T)}
  \item{call}{Command used to call the function}
}
\references{
  See [add paper when available], for more details.
}
\examples{
require(dynSBevo)

## Generation of a random longitudinal network from the dynamic SB model
n <- 100; TT <- 6; k <- 3
# Initial probability vector
piv <- runif(k); piv <- piv/sum(piv)
# Transition probability matrix
Pi <- matrix(0.05, ncol = k, nrow = k); diag(Pi) <- 0.90
# Connection probability matrix
B <- matrix(0.030, ncol = k, nrow = k); diag(B) <- c(0.332, 0.431, 0.229)
# Longitudinal network generation
sample <- SBdyn_draw(n, k, TT, piv, Pi, B)
Y <- sample$Y
V <- sample$V

## Estimation of the dynamic SB model using the standard VEM algorithm - k-means initialization
stdSB.km <- est_dynSB(data = Y, k = 3, start = 2, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, algorithm = "VEM")

## Estimation of the dynamic SB model using the standard VEM algorithm - weighted sbm initializaiton
stdSB.wsb <- est_dynSB(data = Y, k = 3, start = 6, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, algorithm = "VEM")

## Estimation of the dynamic SB model using the evolutionary VEM algorithm
evoSB <- est_dynSB(data = sim$Y, k = k, tol_lk = 1e-6, tol_theta = 1e-6, maxit = 1e3, algorithm = "EVEM", 
                   evolutionaryOptions = list(n_parents = 25, LMP = 0.05, UMP = 0.25, R = 15))
}

