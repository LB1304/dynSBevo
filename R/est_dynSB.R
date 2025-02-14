est_dynSB <- function(data, k, start = NULL, tol_lk = 1e-8, tol_theta = 1e-8, maxit = 1e3, StartVal = NULL, algorithm = c("VEM", "EVEM"), 
                      evolutionaryOptions = list(n_parents = NULL, LMP = NULL, UMP = NULL, R = NULL), Parallel = FALSE, N.threads = NULL) {
  
  if(is.array(data)) {
    Y <- data
  } else {
    stop("Provide data in one of the supported format.")
  }
  
  n <- dim(Y)[1]; TT <- dim(Y)[3]
  
  if (algorithm == "VEM") {
    sv <- switch (start,
                  "1" = Init_Random(n, k, TT), 
                  "2" = Init_Kmeans(Y, n, k, TT),
                  "3" = Init_Kmedoids(Y, n, k, TT),
                  "4" = Init_SpectClust(Y, cl0, n, k, TT),
                  "5" = Init_SBM_Binary(Y, n, k, TT),
                  "6" = Init_SBM_Weighted(Y, n, k, TT),
                  "7" = StartVal
    )
    out <- dynSB_VEM(Y = Y, k = k, sv = sv, tol_lk = tol_lk, tol_theta = tol_theta, maxit = maxit)
  } else if (algorithm == "EVEM") {
    evolutionaryOptions[sapply(evolutionaryOptions, is.null)] <- NULL
    if (length(evolutionaryOptions) != 4) {
      stop("All the evolutionary options must be provided.")
    } else {
      n_parents = evolutionaryOptions$n_parents
      LMP = evolutionaryOptions$LMP
      UMP = evolutionaryOptions$UMP
      R = evolutionaryOptions$R
    }

    if (isTRUE(Parallel)) {
      out <- dynSB_EVEM_PAR(Y = Y, k = k, tol_lk = tol_lk, tol_theta = tol_theta, maxit = maxit, 
                        n_parents = n_parents, LMP = LMP, UMP = UMP, R = R, N.threads = N.threads)
    } else {
      out <- dynSB_EVEM(Y = Y, k = k, tol_lk = tol_lk, tol_theta = tol_theta, maxit = maxit, 
                        n_parents = n_parents, LMP = LMP, UMP = UMP, R = R)
    }
    
  } else {
    stop("Specify an available algorithm.")
  }
  
  out$call = match.call()
  class(out) <- c(class(out), "dynSB")
  return(out)
}
