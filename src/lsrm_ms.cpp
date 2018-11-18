#include <RcppArmadillo.h>
#include <omp.h>

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(openmp)]]

using namespace arma;

//[[Rcpp::export]]
Rcpp::List lsrm_ms_cpp(arma::cube data, arma::vec nsample, arma::vec nitem, const int ndim,
                       const int nset, const int nsamp_max, const int nitem_max, const int ntotal_max,
                       const int niter, const int nburn, const int nthin, const int nprint,
                       const double jump_beta, const double jump_theta, const double jump_z, const double jump_w,
                       const double pr_mean_beta,  const double pr_sd_beta, const double pr_mean_theta, const double pr_sd_theta,
                       const double pr_mean_z, const double pr_sd_z, const double pr_mean_w, const double pr_sd_w, const double pr_z_times,
                       bool option=true, const int cores = 1){
  
  omp_set_num_threads(cores);
  
  int i, j, k, s, accept, count, stsamp, stitem, zstart;
  double num, den, un, ratio, mle;
  double old_like_beta, new_like_beta, old_like_theta, new_like_theta;
  double update_like_sample, update_like_item;
  
  arma::vec ntotal(nset, fill::zeros);
  for(i = 0; i < nset; i++){
    if(i == 0) ntotal(i) = nsample(nset-1) + nsample(i);
    else ntotal(i) = nsample(i-1) + nsample(i);
  }
  
  arma::dmat count_samp(ntotal_max, nset, fill::zeros);
  arma::dmat count_item(nitem_max,  nset, fill::zeros);
  
  for(s = 0; s < nset; s++){
    for(k = 0; k < ntotal(s); k++){
      for(i = 0; i < nitem(s); i++){
        count_samp(k,s) += data(k,i,s);
        count_item(i,s) += data(k,i,s);
      }
    }
  }
  
  arma::dvec old_beta(nitem_max, fill::zeros);
  arma::field<dvec> beta_field(nset, 1);
  for(s = 0; s < nset; s++){
    old_beta.fill(0.0);
    old_beta.subvec(0,nitem(s)-1) = randu<vec>(nitem(s)) * 3.0 - 1.5;
    beta_field(s,0) = old_beta.subvec(0,nitem(s)-1);
  }
  arma::dvec new_beta(nitem_max, fill::zeros);
  
  arma::dvec old_theta(ntotal_max, fill::zeros);
  arma::field<dvec> theta_field(nset, 1);
  for(s = 0; s < nset; s++){
    old_theta.fill(0.0);
    old_theta.subvec(0,ntotal(s)-1) = randu<vec>(ntotal(s)) * 3.0 - 1.5;
    theta_field(s,0) = old_theta.subvec(0,ntotal(s)-1);
  }
  arma::dvec new_theta(ntotal_max, fill::zeros);
  
  arma::dmat old_z(ntotal_max, ndim, fill::randu);
  arma::field<dmat> z_field(nset, 1);
  for(s = 0; s < nset; s++){
    old_z.fill(0.0);
    old_z.submat(0,0,nsample(s)-1,ndim-1) = randu<mat>(nsample(s),ndim) - 0.5;
    z_field(s,0) = old_z.submat(0,0,nsample(s)-1,ndim-1);
  }
  arma::dmat new_z(ntotal_max, ndim, fill::zeros);
  
  arma::dmat old_w(nitem_max, ndim, fill::zeros);
  arma::field<dmat> w_field(nset, 1);
  for(s = 0; s < nset; s++){
    old_w.fill(0.0);
    old_w.submat(0,0,nitem(s)-1,ndim-2) = randu<mat>(nitem(s),ndim-1) - 0.5;
    w_field(s,0) = old_w.submat(0,0,nitem(s)-1,ndim-2);
  }
  arma::dmat new_w(nitem_max, ndim, fill::zeros);
  
  arma::dvec sample_like(ntotal_max, fill::zeros);
  arma::dvec old_sample_dist(ntotal_max, fill::zeros);
  arma::dvec new_sample_dist(ntotal_max, fill::zeros);
  
  arma::dvec item_like(nitem_max, fill::zeros);
  arma::dvec old_item_dist(nitem_max, fill::zeros);
  arma::dvec new_item_dist(nitem_max, fill::zeros);
  
  arma::dcube sample_beta((niter-nburn)/nthin, nitem_max, nset, fill::zeros);
  arma::dcube sample_theta((niter-nburn)/nthin, ntotal_max, nset, fill::zeros);
  arma::dcube sample_z((niter-nburn)/nthin, nsamp_max * ndim, nset, fill::zeros);
  arma::dcube sample_w((niter-nburn)/nthin, nitem_max * ndim, nset, fill::zeros);
  arma::dvec  sample_mle((niter-nburn)/nthin, fill::zeros);
  
  arma::dmat accept_beta(nitem_max, nset, fill::zeros);
  arma::dmat accept_theta(ntotal_max, nset, fill::zeros);
  arma::dmat accept_z(nsamp_max, nset, fill::zeros);
  arma::dmat accept_w(nitem_max, nset, fill::zeros);
  
  arma::dcube distance(ntotal_max, ntotal_max, nset, fill::zeros);
  
  accept = count = 0;
  for(int iter = 1; iter <= niter; iter++){
    for(s = 0; s < nset; s++){
      old_beta.fill(0.0); old_theta.fill(0.0);
      old_z.fill(0.0); old_w.fill(0.0);
      
      old_beta.subvec(0,nitem(s)-1) = beta_field(s,0);
      old_theta.subvec(0,ntotal(s)-1) = theta_field(s,0);
      if(s == 0){
        old_z.submat(0,0,nsample(nset-1)-1,ndim-1) = z_field(nset-1,0);
        old_z.submat(nsample(nset-1),0,nsample(nset-1)+nsample(s)-1,ndim-1) = z_field(s,0);
      }
      else{
        old_z.submat(0,0,nsample(s-1)-1,ndim-1) = z_field(s-1,0);
        old_z.submat(nsample(s-1),0,nsample(s-1)+nsample(s)-1,ndim-1) = z_field(s,0);
      }
      old_w.submat(0,0,nitem(s)-1,ndim-2) = w_field(s,0);
      
      new_beta = old_beta;
      new_theta = old_theta;
      new_z = old_z;
      new_w = old_w;
      
      stsamp = ntotal(s); 
      stitem = nitem(s);
      if(s == 0) zstart = nsample(nset-1);
      else zstart = nsample(s-1);
      
      for(i = 0; i < stitem; i++){
        for(j = 0; j < ndim-1; j++) new_w(i,j) = R::rnorm(old_w(i,j), jump_w);
        sample_like.fill(0.0); 
        old_sample_dist.fill(0.0);
        new_sample_dist.fill(0.0);
        
        #pragma omp parallel for private(j,k) default(shared)
        for(k = 0; k < stsamp; k++){
          for(j = 0; j < ndim; j++){
            old_sample_dist(k) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
            new_sample_dist(k) += std::pow(old_z(k,j)-new_w(i,j), 2.0);
          }
          old_sample_dist(k) = std::sqrt(old_sample_dist(k));
          new_sample_dist(k) = std::sqrt(new_sample_dist(k));
          if(data(k,i,s) == 1.0){
            sample_like(k) -= -std::log(1.0 + std::exp(-(old_beta(i) + old_theta(k) - old_sample_dist(k))));
            sample_like(k) += -std::log(1.0 + std::exp(-(old_beta(i) + old_theta(k) - new_sample_dist(k))));
          }
          else{
            sample_like(k) -= -std::log(1.0 + std::exp(old_beta(i) + old_theta(k) - old_sample_dist(k)));
            sample_like(k) += -std::log(1.0 + std::exp(old_beta(i) + old_theta(k) - new_sample_dist(k)));
          }
        }
        update_like_sample = arma::as_scalar(arma::sum(sample_like));
        
        num = den = 0.0;
        for(j = 0; j < ndim-1; j++){
          num += R::dnorm4(new_w(i,j), pr_mean_w, pr_sd_w, 1);
          den += R::dnorm4(old_w(i,j), pr_mean_w, pr_sd_w, 1);
        }
        ratio = update_like_sample + (num - den);
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0,1);
          if(std::log(un) < ratio) accept = 1;
          else accept = 0;
        }
        
        if(accept == 1){
          for(j = 0; j < ndim-1; j++) old_w(i,j) = new_w(i,j);
          accept_w(i,s) += 1.0 / niter;
        }
        else{
          for(j = 0; j < ndim-1; j++) new_w(i,j) = old_w(i,j);
        }
      }
      
      for(k = zstart; k < stsamp; k++){
        for(j = 0; j < ndim; j++) new_z(k,j) = R::rnorm(old_z(k,j), jump_z);
        item_like.fill(0.0);
        old_item_dist.fill(0.0);
        new_item_dist.fill(0.0);
        
        #pragma omp parallel for private(i,j) default(shared)
        for(i = 0; i < stitem; i++){
          for(j = 0; j < ndim; j++){
            old_item_dist(i) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
            new_item_dist(i) += std::pow(old_z(k,j)-new_w(i,j), 2.0);
          }
          old_item_dist(i) = std::sqrt(old_item_dist(i));
          new_item_dist(i) = std::sqrt(new_item_dist(i));
          if(data(k,i,s) == 1.0){
            item_like(i) -= -std::log(1.0 + std::exp(-(old_beta(i) + old_theta(k) - old_item_dist(i))));
            item_like(i) += -std::log(1.0 + std::exp(-(old_beta(i) + old_theta(k) - new_item_dist(i))));
          }
          else{
            item_like(i) -= -std::log(1.0 + std::exp(old_beta(i) + old_theta(k) - old_item_dist(i)));
            item_like(i) += -std::log(1.0 + std::exp(old_beta(i) + old_theta(k) - new_item_dist(i)));
          }
        }
        update_like_item = arma::as_scalar(arma::sum(item_like));
        
        num = den = 0.0;
        for(j = 0; j < ndim-1; j++){
          num += R::dnorm4(new_z(k,j), pr_mean_z, pr_sd_z, 1);
          den += R::dnorm4(old_z(k,j), pr_mean_z, pr_sd_z, 1);
        }
        num += R::dnorm4(new_z(k,ndim-1), pr_mean_z, pr_sd_z*pr_z_times, 1);
        den += R::dnorm4(old_z(k,ndim-1), pr_mean_z, pr_sd_z*pr_z_times, 1);
        ratio = update_like_item + (num - den);
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0,1);
          if(std::log(un) < ratio) accept = 1;
          else accept = 0;
        }
        
        if(accept == 1){
          for(j = 0; j < ndim; j++) old_z(k,j) = new_z(k,j);
          accept_z((k-zstart),s) += 1.0 / niter;
        }
        else{
          for(j = 0; j < ndim; j++) new_z(k,j) = old_z(k,j);
        }
      }
      
      for(i = 0; i < stitem; i++){
        new_beta(i) = R::rnorm(old_beta(i), jump_beta);
        old_like_beta = new_like_beta = 0.0;
        old_sample_dist.fill(0.0);
        
        #pragma omp parallel for private(k,j) default(shared)
        for(k = 0; k < stsamp; k++){
          for(j = 0; j < ndim; j++) old_sample_dist(k) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
          old_sample_dist(k) = std::sqrt(old_sample_dist(k));
          if(data(k,i,s) == 1.0){
            old_like_beta += -std::log(1.0 + std::exp(-(old_beta(i) + old_theta(k) - old_sample_dist(k))));
            new_like_beta += -std::log(1.0 + std::exp(-(new_beta(i) + old_theta(k) - old_sample_dist(k))));
          }
          else{
            old_like_beta += -std::log(1.0 + std::exp(old_beta(i) + old_theta(k) - old_sample_dist(k)));
            new_like_beta += -std::log(1.0 + std::exp(new_beta(i) + old_theta(k) - old_sample_dist(k)));
          }
        }
        
        num = R::dnorm4(new_beta(i), pr_mean_beta, pr_sd_beta, 1);
        den = R::dnorm4(old_beta(i), pr_mean_beta, pr_sd_beta, 1);
        ratio = (new_like_beta-old_like_beta) + (num-den);
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0,1);
          if(std::log(un) < ratio) accept = 1;
          else accept = 0;
        }
        
        if(accept == 1){
          old_beta(i) = new_beta(i);
          accept_beta(i,s) += 1.0 / niter;
        }
        else{
          new_beta(i) = old_beta(i);
        }
      }
      
      for(k = 0; k < stsamp; k++){
        new_theta(k) = R::rnorm(old_theta(k), jump_theta);
        old_like_theta = new_like_theta = 0.0;
        old_item_dist.fill(0.0);
        
        #pragma omp parallel for private(i,j) default(shared)
        for(i = 0; i < stitem; i++){
          for(j = 0; j < ndim; j++) old_item_dist(i) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
          old_item_dist(i) = std::sqrt(old_item_dist(i));
          if(data(k,i,s) == 1.0){
            old_like_theta += -std::log(1.0 + std::exp(-(old_beta(i) + old_theta(k) - old_item_dist(i))));
            new_like_theta += -std::log(1.0 + std::exp(-(old_beta(i) + new_theta(k) - old_item_dist(i))));
          }
          else{
            old_like_theta += -std::log(1.0 + std::exp(old_beta(i) + old_theta(k) - old_item_dist(i)));
            new_like_theta += -std::log(1.0 + std::exp(old_beta(i) + new_theta(k) - old_item_dist(i)));
          }
        }
        
        num = den = 0.0;
        for(j = 0; j < ndim; j++){
          num += R::dnorm4(new_theta(k), pr_mean_theta, pr_sd_theta, 1);
          den += R::dnorm4(old_theta(k), pr_mean_theta, pr_sd_theta, 1);
        }
        ratio = (new_like_theta-old_like_theta) + (num-den);
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0,1);
          if(std::log(un) < ratio) accept = 1;
          else accept = 0;
        }
        
        if(accept == 1){
          old_theta(k) = new_theta(k);
          accept_theta(k,s) += 1.0 / niter;
        }
        else new_theta(k) = old_theta(k);
      }
      
      if(iter > nburn && iter % nthin == 0){
        if(s == 0){
          mle = 0.0; 
          distance.fill(0.0);
        } 
        #pragma omp parallel for private(i,j,k) default(shared)
        for(k = 0; k < stsamp; k++){
          for(i = 0; i < stitem; i++){
            for(j = 0; j < ndim; j++) distance(k,i,s) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
            distance(k,i,s) = std::sqrt(distance(k,i,s));
            if(data(k,i,s) == 1.0) mle += -std::log(1.0 + std::exp(-(old_beta(i) + old_theta(k) - distance(k,i,s))));
            else mle += -std::log(1.0 + std::exp(old_beta(i) + old_theta(k) - distance(k,i,s)));
          }
        }
        
        for(k = 0; k < stsamp; k++) mle += R::dnorm4(old_theta(k), pr_mean_theta, pr_sd_theta, 1);
        for(i = 0; i < stitem; i++) mle += R::dnorm4(old_beta(i), pr_mean_beta, pr_sd_beta, 1);
        for(k = 0; k < stsamp; k++){
          for(j = 0; j < ndim-1; j++) mle += R::dnorm4(old_z(k,j), pr_mean_z, pr_sd_z, 1);
          mle += R::dnorm4(old_z(k,ndim-1), pr_mean_z, pr_sd_z*pr_z_times, 1);
        }
        for(i = 0; i < stitem; i++){
          for(j = 0; j < ndim-1; j++) mle += R::dnorm4(old_w(i,j), pr_mean_w, pr_sd_w, 1);
        }
        
        for(k = zstart; k < stsamp; k++)
          for(j = 0; j < ndim; j++) sample_z(count,((k-zstart)*ndim+j),s) = old_z(k,j);
        for(i = 0; i < stitem; i++)
          for(j = 0; j < ndim; j++) sample_w(count,(i*ndim+j),s) = old_w(i,j);
        for(i = 0; i < stitem; i++) sample_beta(count,i,s) = old_beta(i);
        for(k = 0; k < stsamp; k++) sample_theta(count,k,s) = old_theta(k);
        sample_mle(count) = mle;
      }
      
      if(iter % nprint == 0){
        if(option){
          printf("%.5d-SET%.2d ", iter, s);
          for(i = 0; i < nitem(s); i++) printf("% .3f ", old_beta(i));
          printf("\n");
        }
      }
      
      beta_field(s,0) = old_beta.subvec(0,nitem(s)-1);
      theta_field(s,0) = old_theta.subvec(0,ntotal(s)-1);
      z_field(s,0) = old_z.submat(nsample(nset-1),0,nsample(nset-1)+nsample(s)-1,ndim-1);
      w_field(s,0) = old_w.submat(0,0,nitem(s)-1,ndim-2);
    }
    
    if(iter > nburn && iter % nthin == 0){
      count++;
    }
  }
  
  Rcpp::List output;
  output["beta"] = sample_beta;
  output["theta"] = sample_theta;
  output["z"] = sample_z;
  output["w"] = sample_w;
  output["accept_beta"] = accept_beta;
  output["accept_theta"] = accept_theta;
  output["accept_z"] = accept_z;
  output["accept_w"] = accept_w;
  output["posterior"] = sample_mle;
  
  return(output);
}