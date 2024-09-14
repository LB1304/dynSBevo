SBdyn_draw <- function(n, k, TT, piv, Pi, B) {
  
  # Cluster allocation
  V <- matrix(0, n, TT)
  V[, 1] <- sample(x = 1:k, size = n, replace = TRUE, prob = piv)
  for(t in 2:TT){
    for(i in 1:n) {
      V[i, t] <- sample(x = 1:k, size = 1, prob = Pi[V[i, t-1], ])
    }
  }	
  
  # Dynamic network
  Y <- array(0, c(n, n, TT))
  for(t in 1:TT) for(i in 1:(n-1)) for(j in (i+1):n) {
    prob <- B[V[i, t], V[j, t]]
    exist.edge <- rbinom(n = 1, size = 1, prob = prob)
    if (exist.edge == 1) {
      Y[i, j, t] <- Y[j, i, t] <- 1
    }
  }
  
  out <- list(Y = Y, V = V)
}