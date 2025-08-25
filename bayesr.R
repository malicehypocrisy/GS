#############################
## Author: Jian Zeng
## Date: 16 June 2022
#############################

library(MCMCpack)

bayesr = function(X, y, niter = 1000, gamma = c(0, 0.01, 0.1, 1), startPi = c(0.5, 0.3, 0.15, 0.05), startH2 = 0.5){
  n = nrow(X)          # number of observations
  m = ncol(X)          # number of SNPs
  ndist = length(startPi)  # number of mixture distributions
  pi = startPi         # starting value for pi
  h2 = startH2         # starting value for heritability
  vary = var(y)        # phenotypic variance
  varg = vary*h2       # starting value for genetic variance
  vare = vary*(1-h2)   # starting value for residual variance
  sigmaSq = varg/(m*sum(gamma*pi))    # common factor of SNP effect variance
  nub = 4              # prior degrees of freedom for SNP effect variance
  nue = 4              # prior degrees of freedom for residual variance
  scaleb = (nub-2)/nub*sigmaSq  # prior scale parameter for SNP effect variance
  scalee = (nue-2)/nue*vare     # prior scale parameter for residual variance
  beta = array(0,m)    # vector of SNP effects
  beta_mcmc = matrix(0,niter,m) # MCMC samples of SNP effects
  mu = mean(y)         # overall mean
  ycorr = y - mu       # adjusted y  
  xpx = apply(X, 2, crossprod)  ## diagonal elements of X'X
  probDelta = vector("numeric", ndist)
  logDelta = array(0,2)
  keptIter = NULL
  
  for (iter in 1:niter){
    # sampling intercept
    ycorr = ycorr + mu
    rhs = sum(ycorr)
    invLhs = 1/n
    muHat = invLhs*rhs
    mu = rnorm(1, muHat, sqrt(invLhs*vare)) 
    ycorr = ycorr - mu

    # sampling SNP effects
    logPi = log(pi)
    logPiComp = log(1-pi)
    invSigmaSq = 1/(gamma*c(sigmaSq))
    logSigmaSq = log(gamma*c(sigmaSq))
    nsnpDist = rep(0, ndist)
    ssq = 0
    nnz = 0
    ghat = array(0,n)
    for (j in 1:m){
      oldSample = beta[j]
      rhs = crossprod(X[,j], ycorr) + xpx[j]*oldSample
      rhs = rhs/vare
      invLhs = 1.0/(xpx[j]/c(vare) + invSigmaSq)
      betaHat = invLhs*c(rhs)
      
      # sampling mixture distribution membership
      logDelta = 0.5*(log(invLhs) - logSigmaSq + betaHat*c(rhs)) + logPi
      logDelta[1] = logPi[1];
      for (k in 1:ndist) {
        probDelta[k] = 1.0/sum(exp(logDelta - logDelta[k]))
      }
      
      delta = sample(1:ndist, 1, prob = probDelta)
      nsnpDist[delta] = nsnpDist[delta] + 1
      
      if (delta > 1) {
        beta[j] = rnorm(1, betaHat[delta], sqrt(invLhs[delta]))
        ycorr = ycorr + X[,j]*(oldSample - beta[j])
        ghat = ghat + X[,j]*beta[j]
        ssq = ssq + beta[j]^2 / gamma[delta]
        nnz = nnz + 1
      } else {
        if (oldSample) ycorr = ycorr + X[,j]*oldSample
        beta[j] = 0
      }
    }	
    beta_mcmc[iter,] = beta
    
    # sampling pi
    pi = rdirichlet(1, nsnpDist + 1)
    
    # sampling SNP effect variance
    sigmaSq = (ssq + nub*scaleb)/rchisq(1, nnz+nub)
    
    # sampling residual variance
    vare = (crossprod(ycorr) + nue*scalee)/rchisq(1, n+nue)
    
    # compute heritability
    varg = var(ghat)
    h2  = varg/(varg + vare)

    keptIter <- rbind(keptIter,c(pi, nnz, sigmaSq, h2, vare, varg))
    
    if (!(iter%%100)) {
      cat (sprintf("\n iter %4s, nnz = %4s, sigmaSq = %6.3f, h2 = %6.3f, vare = %6.3f, varg = %6.3f \n", iter, nnz, sigmaSq, h2, vare, varg))
    }
  }
  
  colnames(keptIter) <- c(paste0("Pi", 1:length(pi)),"Nnz","SigmaSq","h2","Vare","Varg")
  postMean = apply(keptIter, 2, mean)
  cat("\nPosterior mean:\n")
  print(postMean)
  return(list(par=keptIter, beta=beta_mcmc))
}