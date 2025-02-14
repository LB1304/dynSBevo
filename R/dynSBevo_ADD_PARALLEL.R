dynSB_EVEM_PAR <- function (Y, k, tol_lk, tol_theta, maxit, n_parents, LMP, UMP, R, N.threads = 8) {
  n <- dim(Y)[2]
  TT <- dim(Y)[3]
  
  # 1. Initial values
  PV0_Ran <- dynSB_Random_Initialization_step(Y, n, k, TT, n_parents)
  PV0_Det1 <- dynSB_Kmeans_Initialization_step(Y, n, k, TT, n_parents, LMP, UMP)
  PV0_Det2 <- dynSB_SBM_Initialization_step(Y, n, k, TT, n_parents, LMP, UMP)

  ### Parallelization
  # cl <- makeCluster(N.threads, type = "FORK")    # --> Linux/Mac
  cl <- parallel::makeCluster(N.threads, type = "PSOCK")   # --> Windows/Linux/Mac
  parallel::clusterEvalQ(cl, library(dynSBevo))
  parallel::clusterExport(cl, varlist = c("Y"), envir = environment())
  doParallel::registerDoParallel(cl)

  # 2. Evolution (separate)
  PV1_Ran_Aux <- dynSB_EVEM_step_PAR(Y, k, PV0_Ran, tol_lk, tol_theta, n_parents, LMP, UMP, R, n, TT, FALSE)
  fit1_Ran <- PV1_Ran_Aux$J
  PV1_Ran <- PV1_Ran_Aux$PV
  # cat("Evolution Random.\n")
  PV1_Det1_Aux <- dynSB_EVEM_step_PAR(Y, k, PV0_Det1, tol_lk, tol_theta, n_parents, LMP, UMP, R, n, TT, TRUE)
  fit1_Det1 <- PV1_Det1_Aux$J
  PV1_Det1 <- PV1_Det1_Aux$PV
  # cat("Evolution Deterministic (K-means).\n")
  PV1_Det2_Aux <- dynSB_EVEM_step_PAR(Y, k, PV0_Det2, tol_lk, tol_theta, n_parents, LMP, UMP, R, n, TT, TRUE)
  fit1_Det2 <- PV1_Det2_Aux$J
  PV1_Det2 <- PV1_Det2_Aux$PV
  # cat("Evolution Deterministic (SBM).\n")
  
  # 3. Select best
  PV2 <- dynSB_TotalSelection_step(PV1_Ran, PV1_Det1, PV1_Det2, fit1_Ran, fit1_Det1, fit1_Det2)
  # cat("Selection.\n")
  
  # 4. Evolution (in common)
  PV3_Aux <- dynSB_EVEM_step_PAR(Y, k, PV2, tol_lk, tol_theta, n_parents, LMP, UMP, R, n, TT, FALSE)
  fit3 <- PV3_Aux$J
  PV3 <- PV3_Aux$PV
  # cat("Evolution Common.\n")
  
  # 5. Last EM step
  ind_best <- which.max(fit3)
  PV_best <- PV3[[ind_best]]
  out_best <- dynSB_LastUpdate_step(Y, k, PV_best, tol_lk, tol_theta, maxit)
  # cat("Last Step.\n")
  
  parallel::stopCluster(cl)
  return(out_best)
}




dynSB_EVEM_step_PAR <- function(Y, k, PV1, tol_lk, tol_theta, n_parents, LMP, UMP, R, n, TT, keep_best) {
  it <- 0
  alt <- FALSE
  
  while (!alt) {
    it <- it+1
    
    # 1. Update parents and compute fitness (Parallelized)
    PV2 <- foreach::foreach(b = 1:n_parents) foreach::`%dopar%` {
      PV1_sub1 <- PV1[[b]]
      dynSBevo::dynSB_Update_step(Y, k, PV1_sub1, tol_lk, tol_theta, R)
    }
    fit2 <- unlist(lapply(PV2, "[[", "J"))
    # 2. Mutation
    PV3 <- foreach::foreach(b = 1:n_parents) foreach::`%dopar%` {
      PV2_sub1 <- PV2[[b]]
      Tau1 <- PV2_sub1$Tau1
      TAU <- PV2_sub1$TAU
      dynSBevo::dynSB_Mutation_step(Tau1, TAU, n, k, TT, LMP, UMP);
    }
    if (keep_best) {
      PV3[[1]] <- PV2[[1]]
    }
    # 3. Update children and compute fitness (Parallelized)
    PV4 <- foreach::foreach(b = 1:n_parents) foreach::`%dopar%` {
      PV3_sub1 <- PV3[[b]]
      dynSBevo::dynSB_Update_step(Y, k, PV3_sub1, tol_lk, tol_theta, R)
    }
    fit4 <- unlist(lapply(PV4, "[[", "J"))
    # 4. Select new parents
    PV5 <- dynSB_Selection_step(PV2, PV4, fit2, fit4, keep_best)
    fit5 <- unlist(lapply(PV5, "[[", "J"))
    
    if (it > 1) {
      alt <- (abs(max(fit5) - fit_old)/abs(fit_old) < tol_lk)
    }
    fit_old <- max(fit5)
    PV1 <- PV5
  }
  
  return(list(PV = PV5, J = fit5))
}
