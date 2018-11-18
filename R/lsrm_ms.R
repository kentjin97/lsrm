#' title
#' 
#' 
#' @export
lsrm_ms <- function(dataset, nsample, nitem, ndim = 3, nset, nsamp_max, nitem_max, ntotal_max, 
                    niter = 30000, nburn = 5000, nthin = 5, nprint = 100,
                    jump_beta = 0.2, jump_theta = 1.0, jump_z = 0.05, jump_w = 0.05,
                    pr_mean_beta = 0.0, pr_sd_beta = 10.0, pr_mean_theta = 0.0, pr_sd_theta = 10.0,
                    pr_mean_z = 0.0, pr_sd_z = 1.0, pr_mean_w = 0.0, pr_sd_w = 1.0, pr_z_times = 4.0,
                    option = TRUE, latent_equal = TRUE, cores = 1){
  
  if((niter - nburn) %% nthin == 0){
    if(!latent_equal){
      output = lsrm_ms_cpp(dataset, nsample, nitem, ndim, 
                           nset, nsamp_max, nitem_max, ntotal_max,
                           niter, nburn, nthin, nprint,
                           jump_beta, jump_theta, jump_z, jump_w,
                           pr_mean_beta, pr_sd_beta, pr_mean_theta, pr_sd_theta,
                           pr_mean_z, pr_sd_z, pr_mean_w, pr_sd_z, pr_z_times,
                           option, cores)
    }
    if(latent_equal){
      output = lsrm_ms_eq_cpp(dataset, nsample, nitem, ndim, 
                              nset, nsamp_max, nitem_max, ntotal_max,
                              niter, nburn, nthin, nprint,
                              jump_beta, jump_theta, jump_z, jump_w,
                              pr_mean_beta, pr_sd_beta, pr_mean_theta, pr_sd_theta,
                              pr_mean_z, pr_sd_z, pr_mean_w, pr_sd_z, pr_z_times,
                              option, cores)
    }
    
    nmcmc = as.integer((niter - nburn) / nthin)
    max.address = which.max(output$posterior)
    z.proc = array(0,dim=c(nsamp_max,ndim,nmcmc,nset))
    w.proc = array(0,dim=c(nitem_max,ndim,nmcmc,nset))
    for(s in 1:nset){
      z.star = matrix(output$z[max.address,(1:(nsample[s]*ndim)),s],ncol=ndim,byrow=TRUE)
      w.star = matrix(output$w[max.address,(1:(nitem[s]*ndim)),s],ncol=ndim,byrow=TRUE)
      for(iter in 1:nmcmc){
        z.iter = matrix(output$z[iter,(1:(nsample[s]*ndim)),s],ncol=ndim,byrow=TRUE)
        if(iter != max.address) z.proc[(1:nsample[s]),,iter,s] = procrustes(z.iter,z.star)$X.new
        else z.proc[(1:nsample[s]),,iter,s] = z.iter
        
        w.iter = matrix(output$w[iter,(1:(nitem[s]*ndim)),s],ncol=ndim,byrow=TRUE)
        if(iter != max.address) w.proc[(1:nitem[s]),,iter,s] = procrustes(w.iter,w.star)$X.new
        else w.proc[(1:nitem[s]),,iter,s] = w.iter
      }
    }
    
    beta.est = matrix(0,nitem_max,nset)
    beta.std = matrix(0,nitem_max,nset)
    theta.est = matrix(0,nsamp_max,nset)
    theta.std = matrix(0,nsamp_max,nset)
    for(s in 1:nset){
      for(i in 1:nitem[s]){
        beta.est[i,s] = mean(output$beta[,i,s])  
        beta.std[i,s] = sd(output$beta[,i,s])
      }
    }
    for(s in 1:nset){
      for(k in 1:nsample[s]){
        theta.est[k,s] = mean(output$theta[,k,s])
        theta.std[k,s] = sd(output$theta[,k,s])
      }
    }
    
    w.est = array(NA,dim=c(nitem_max,ndim,nset))
    for(s in 1:nset){
      for(i in 1:nitem[s]){
        for(j in 1:ndim){
          w.est[i,j,s] = mean(w.proc[i,j,,s])
        }
      }
    }
    
    z.est = array(NA,dim=c(nsamp_max,ndim,nset))
    for(s in 1:nset){
      for(k in 1:nsample[s]){
        for(j in 1:ndim){
          z.est[k,j,s] = mean(z.proc[k,j,,s])
        }
      }
    }
    
    return(list(beta=output$beta, theta=output$theta, z=z.proc, w=w.proc, 
                beta.estimate=beta.est, theta.estimate=theta.est, 
                beta.se=beta.std, theta.se=theta.std,
                z.estimate=z.est, w.estimate=w.est, 
                accept_beta=output$accept_beta, 
                accept_theta=output$accept_theta,
                accept_z=output$accept_z,
                accept_w=output$accept_w))
  }
  else{
    print("Error: The total size of MCMC sample is not integer")
    return(-999)
  }
}