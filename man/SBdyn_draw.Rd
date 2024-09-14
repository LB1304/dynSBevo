\name{SBdyn_draw}
\alias{SBdyn_draw}
\title{Generation of random samples from the dynamic stochastic block model for longitudinal network data}
\description{
This function samples a random longitudinal network data from the dynamic stochastic block model.
}
\usage{
SBdyn_draw(n, k, TT, piv, Pi, B)
}
\arguments{
  \item{n}{Number of nodes}
  \item{k}{An integer specifying the number of latent blocks}
  \item{TT}{Number of time occasions}
  \item{piv}{Initial probability vector (k)}
  \item{Pi}{Transition probability matrix (k x k)}
  \item{B}{Connection probability matrix (k x k)}
}
\value{
  \item{Y}{Adjacency \code{array} of the longitudinal network}
  \item{U}{\code{Matrix} of the latent block for each node at each time occasion (n x k)}
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
}
\author{Luca Brusa, Fulvia Pennoni}

