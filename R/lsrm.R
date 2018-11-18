#' title
#'
#'
#' @export
lsrm <- function(data, nsample, nitem, ndim = 3, niter = 30000, nburn = 5000, nthin = 5, nprint = 100,
                 jump_beta = 0.1, jump_theta = 0.5, jump_z = 0.05, jump_w = 0.05,
                 pr_mean_beta = 0.0, pr_sd_beta = 10.0, pr_mean_theta = 0.0, pr_sd_theta = 10.0,
                 pr_mean_z = 0.0, pr_sd_z = 1.0, pr_mean_w = 0.0, pr_sd_w = 1.0, pr_z_times = 4.0,
                 option = TRUE, dist_option = 1, cores = 1){
  
  if((niter - nburn) %% nthin == 0){
    if(dist_option == 0){
      output = lsrm_min_cpp(data, nsample, nitem, ndim, niter, nburn, nthin, nprint,
                            jump_beta, jump_theta, jump_z, jump_w, 
                            pr_mean_beta, pr_sd_beta, pr_mean_theta, pr_sd_theta,
                            pr_mean_z, pr_sd_z, pr_mean_w, pr_sd_w, option, cores)
    }
    if(dist_option == 2){
      output = lsrm_norm_cpp(data, nsample, nitem, ndim, niter, nburn, nthin, nprint,
                             jump_beta, jump_theta, jump_z, jump_w, 
                             pr_mean_beta, pr_sd_beta, pr_mean_theta, pr_sd_theta,
                             pr_mean_z, pr_sd_z, pr_mean_w, pr_sd_w, pr_z_times, option, cores)
    }
    if(dist_option == 1){
      output = lsrm_norm_eq_cpp(data, nsample, nitem, ndim, niter, nburn, nthin, nprint,
                                jump_beta, jump_theta, jump_z, jump_w, 
                                pr_mean_beta, pr_sd_beta, pr_mean_theta, pr_sd_theta,
                                pr_mean_z, pr_sd_z, pr_mean_w, pr_sd_w, option, cores)
    }
    
    nmcmc = as.integer((niter - nburn) / nthin)
    max.address = which.max(output$posterior)
    w.star = output$w[max.address,,]
    z.star = output$z[max.address,,]
    w.proc = array(0,dim=c(nmcmc,nitem,ndim))
    z.proc = array(0,dim=c(nmcmc,nsample,ndim))
    
    for(iter in 1:nmcmc){
      z.iter = output$z[iter,,]
      w.iter = output$w[iter,,]
      if(iter != max.address){
        w.proc[iter,,] = procrustes(w.iter,w.star)$X.new
        z.proc[iter,,] = procrustes(z.iter,z.star)$X.new
      }
      else{
        w.proc[iter,,] = w.iter
        z.proc[iter,,] = z.iter
      }
    }
    
    beta.est = apply(output$beta,2,mean)
    beta.std = apply(output$beta,2,sd)
    theta.est = apply(output$theta,2,mean)
    theta.std = apply(output$theta,2,sd)
    
    w.est = matrix(NA,nitem,ndim)
    z.est = matrix(NA,nsample,ndim)
    for(i in 1:nitem){
      for(j in 1:ndim){
        w.est[i,j] = mean(w.proc[,i,j])
      }
    }
    for(k in 1:nsample){
      for(j in 1:ndim){
        z.est[k,j] = mean(z.proc[,k,j])
      }
    }
    
    if(dist_option == 2){
      return(list(beta=output$beta, theta=output$theta, z=z.proc, w=w.proc[,,(1:(ndim-1))], 
                  beta.estimate=beta.est, theta.estimate=theta.est, z.estimate=z.est, w.estimate=w.est[,(1:(ndim-1))],
                  beta.se=beta.std, theta.se=theta.std,
                  accept_beta=output$accept_beta, accept_theta=output$accept_theta,
                  accept_z=output$accept_z, accept_w=output$accept_w))
    }
    else{
      return(list(beta=output$beta, theta=output$theta, z=z.proc, w=w.proc, 
                  beta.estimate=beta.est, theta.estimate=theta.est, z.estimate=z.est, w.estimate=w.est,
                  beta.se=beta.std, theta.se=theta.std,
                  accept_beta=output$accept_beta, accept_theta=output$accept_theta,
                  accept_z=output$accept_z, accept_w=output$accept_w))
    }
    
  }
  else{
    print("Error: The total size of MCMC sample is not integer")
    return(-999)
  }
  
}