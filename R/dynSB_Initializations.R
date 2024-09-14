## Initialization strategies

# 1. Random
Init_Random <- function(n, k, TT) {
  Tau1 = matrix(runif(n*k), n, k)
  Tau1 = (1/rowSums(Tau1))*Tau1
  TAU = array(runif(n*k*TT), c(n, k, k, TT))
  TAU[, , , 1] = 0
  for(i in 1:n) for(t in 1:TT) TAU[i, , , t] = (1/rowSums(TAU[i, , , t]))*TAU[i, , , t]
  
  TAU_new <- Create_Field(N = TT, d1 = n, d2 = k, d3 = k)
  for (t in 1:TT) {
    TAU_new <- Update_Field(Field = TAU_new, New_Slice = TAU[, , , t], n = t-1)
  }
  
  sv <- dynSB_compute_derivatives(Tau1, TAU_new, n, k, TT)
  
  return(sv)
}


# 2. K-means on all observation jointly
Init_Kmeans <- function(Y, n, k, TT) {
  for(t in 1:TT) diag(Y[,,t]) = 0
  YY = Y[,,1]
  for(t in 2:TT) YY = rbind(YY,Y[,,t])
  cl0 = matrix(kmeans(YY, k, nstart = 10)$cl, n, TT)
  Tau1 = matrix(0, n, k)
  for(i in 1:n) Tau1[i, cl0[i, 1]] = 1
  TAU = array(0, c(n, k, k, TT))
  for(i in 1:n) for(t in 2:TT) TAU[i, , cl0[i, t], t] = 1
  
  TAU_new <- Create_Field(N = TT, d1 = n, d2 = k, d3 = k)
  for (t in 1:TT) {
    TAU_new <- Update_Field(Field = TAU_new, New_Slice = TAU[, , , t], n = t-1)
  }
  
  sv <- dynSB_compute_derivatives(Tau1, TAU_new, n, k, TT)
  
  return(sv)
}


# 3. K-medoids on all observation jointly
Init_Kmedoids <- function(Y, n, k, TT) {
  for(t in 1:TT) diag(Y[, , t]) = 0
  YY = Y[,,1]
  for(t in 2:TT) YY = rbind(YY,Y[,,t])
  cl0 = matrix(cluster::pam(YY, k, nstart = 10)$clustering, n, TT)
  Tau1 = matrix(0, n, k)
  for(i in 1:n) Tau1[i,cl0[i,1]] = 1
  TAU = array(0,c(n, k, k, TT))
  for(i in 1:n) for(t in 2:TT) TAU[i, , cl0[i, t], t] = 1
  
  TAU_new <- Create_Field(N = TT, d1 = n, d2 = k, d3 = k)
  for (t in 1:TT) {
    TAU_new <- Update_Field(Field = TAU_new, New_Slice = TAU[, , , t], n = t-1)
  }
  
  sv <- dynSB_compute_derivatives(Tau1, TAU_new, n, k, TT)
  
  return(sv)
}


# 4. Spectral clustering on all observation jointly 
Init_SpectClust <- function(Y, cl0, n, k, TT) {
  for(t in 2:TT) cl0 = cbind(cl0, matrix(kmeans(Y[, , t], k, nstart = 1)$cl, n, 1))
  Tau1 = matrix(0, n, k)
  for(i in 1:n) Tau1[i, cl0[i, 1]] = 1
  TAU = array(0, c(n, k, k, TT))
  for(i in 1:n) for(t in 2:TT) TAU[i, , cl0[i, t], t] = 1
  
  TAU_new <- Create_Field(N = TT, d1 = n, d2 = k, d3 = k)
  for (t in 1:TT) {
    TAU_new <- Update_Field(Field = TAU_new, New_Slice = TAU[, , , t], n = t-1)
  }
  
  sv <- dynSB_compute_derivatives(Tau1, TAU_new, n, k, TT)
  
  return(sv)
}
## This initialization strategy requires the computation of the "Global Spectral Clustering" algorithm
## defined in "Liu F., Choi D., Xie L., Roeder K. (2017) Global spectral clustering in dynamic networks"
## (MATLAB code available at the author's GitHub page: https://github.com/letitiaLiu/PisCES)


# 5. Stochastic block model on the aggregated binarized network
Init_SBM_Binary <- function(Y, n, k, TT) {
  Y_aggr <- matrix(0, n, n)
  for (t in 1:TT) {
    Y_aggr <- Y_aggr + Y[, , t]
  }
  Y_aggr <- ifelse(Y_aggr > 0, 1, 0)
  est <- sbm:::estimateSimpleSBM(netMat = Y_aggr, model = "bernoulli", 
                                 estimOptions = list(verbosity = 0, plot = FALSE, exploreMin = k, exploreMax = k))
  est$setModel(k)
  cl0 <- matrix(est$memberships, nrow = n, ncol = TT, byrow = FALSE)
  Tau1 = matrix(0, n, k)
  for(i in 1:n) Tau1[i, cl0[i, 1]] = 1
  TAU = array(0, c(n, k, k, TT))
  for(i in 1:n) for(t in 2:TT) TAU[i, , cl0[i, t], t] = 1
  
  TAU_new <- Create_Field(N = TT, d1 = n, d2 = k, d3 = k)
  for (t in 1:TT) {
    TAU_new <- Update_Field(Field = TAU_new, New_Slice = TAU[, , , t], n = t-1)
  }
  
  sv <- dynSB_compute_derivatives(Tau1, TAU_new, n, k, TT)
  
  return(sv)
}


# 6. Stochastic block model on the aggregated weighted network
Init_SBM_Weighted <- function(Y, n, k, TT) {
  Y_aggr <- matrix(0, n, n)
  for (t in 1:TT) {
    Y_aggr <- Y_aggr + Y[, , t]
  }
  est <- sbm:::estimateSimpleSBM(netMat = Y_aggr, model = "poisson", 
                                 estimOptions = list(verbosity = 0, plot = FALSE, exploreMin = k, exploreMax = k))
  est$setModel(k)
  cl0 <- matrix(est$memberships, nrow = n, ncol = TT, byrow = FALSE)
  Tau1 = matrix(0, n, k)
  for(i in 1:n) Tau1[i, cl0[i, 1]] = 1
  TAU = array(0, c(n, k, k, TT))
  for(i in 1:n) for(t in 2:TT) TAU[i, , cl0[i, t], t] = 1
  
  TAU_new <- Create_Field(N = TT, d1 = n, d2 = k, d3 = k)
  for (t in 1:TT) {
    TAU_new <- Update_Field(Field = TAU_new, New_Slice = TAU[, , , t], n = t-1)
  }
  
  sv <- dynSB_compute_derivatives(Tau1, TAU_new, n, k, TT)
  
  return(sv)
}


# 7. Fixed starting values
Init_Fixed <- function(StartVal, n, k, TT) {
  Tau1 = StartVal$Tau1
  TAU <- Create_Field(N = TT, d1 = n, d2 = k, d3 = k)
  for (t in 1:TT) {
    TAU <- Update_Field(Field = TAU, New_Slice = StartVal$TAU[, , , t], n = t-1)
  }
  
  sv <- dynSBevo::dynSB_compute_derivatives(Tau1 = Tau1, TAU = TAU, n = n, k = k, TT = TT)
  
  return(sv)
}




