#include <RcppArmadillo.h>
#include <omp.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace arma;

//[[Rcpp::export]]
Rcpp::List lsrm_norm_eq_cpp(arma::mat data, const int nsample, const int nitem, const int ndim,
                         const int niter, const int nburn, const int nthin, const int nprint,  
                         const double jump_beta, const double jump_theta, const double jump_z, const double jump_w,
                         const double pr_mean_beta,  const double pr_sd_beta, const double pr_mean_theta, const double pr_sd_theta,
                         const double pr_mean_z, const double pr_sd_z, const double pr_mean_w, const double pr_sd_w,
                         bool option=true, const int cores = 1){
  
  omp_set_num_threads(cores);
  
  int i, j, k, count, accept;
  double num, den, un, ratio, mle;
  double old_like_beta, new_like_beta, old_like_theta, new_like_theta;
  double update_like_sample, update_like_item;
  
  arma::dvec old_beta(nitem, fill::randu);
  old_beta = 3.0 * old_beta - 1.5;
  arma::dvec new_beta = old_beta;
  
  arma::dvec old_theta(nsample, fill::randu);
  old_theta = 3.0 * old_theta - 1.5;
  arma::dvec new_theta = old_theta;
  
  arma::dmat old_z(nsample, ndim, fill::randu);
  old_z -= 0.5;
  arma::dmat new_z = old_z;
  
  arma::dmat old_w(nitem, ndim, fill::randu);
  old_w -= 0.5;
  arma::dmat new_w = old_w;
  
  arma::dmat  sample_beta((niter-nburn)/nthin, nitem, fill::zeros);
  arma::dmat  sample_theta((niter-nburn)/nthin, nsample, fill::zeros);
  arma::dcube sample_z((niter-nburn)/nthin, nsample, ndim, fill::zeros);
  arma::dcube sample_w((niter-nburn)/nthin, nitem, ndim, fill::zeros);
  arma::dvec  sample_mle((niter-nburn)/nthin, fill::zeros);
  
  arma::dvec accept_beta(nitem, fill::zeros);
  arma::dvec accept_theta(nsample, fill::zeros);
  arma::dvec accept_z(nsample, fill::zeros);
  arma::dvec accept_w(nitem, fill::zeros);
  
  arma::dvec sample_like(nsample, fill::zeros);
  arma::dvec old_sample_dist(nsample, fill::zeros);
  arma::dvec new_sample_dist(nsample, fill::zeros);
  
  arma::dvec item_like(nitem, fill::zeros);
  arma::dvec old_item_dist(nitem, fill::zeros);
  arma::dvec new_item_dist(nitem, fill::zeros);
  
  arma::dmat distance(nsample, nitem, fill::zeros);
  
  // index k: represents sample
  // index i: represents item
  // index j: represents dimension
  
  count = 0;
  for(int iter = 1; iter <= niter; iter++){
    
    for(i = 0; i < nitem; i++){
      for(j = 0; j < ndim; j++) new_w(i,j) = R::rnorm(old_w(i,j), jump_w);
      sample_like.fill(0.0); 
      old_sample_dist.fill(0.0);
      new_sample_dist.fill(0.0);
      
      #pragma omp parallel for private(j,k) default(shared)
      for(k = 0; k < nsample; k++){
        for(j = 0; j < ndim; j++){
          old_sample_dist(k) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
          new_sample_dist(k) += std::pow(old_z(k,j)-new_w(i,j), 2.0);
        }
        old_sample_dist(k) = std::sqrt(old_sample_dist(k));
        new_sample_dist(k) = std::sqrt(new_sample_dist(k));
        if(data(k,i) == 1.0){
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
      for(j = 0; j < ndim; j++){
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
        for(j = 0; j < ndim; j++) old_w(i,j) = new_w(i,j);
        accept_w(i) += 1.0 / niter;
      }
      else{
        for(j = 0; j < ndim; j++) new_w(i,j) = old_w(i,j);
      }
    }
    
    for(k = 0; k < nsample; k++){
      for(j = 0; j < ndim; j++) new_z(k,j) = R::rnorm(old_z(k,j), jump_z);
      item_like.fill(0.0);
      old_item_dist.fill(0.0);
      new_item_dist.fill(0.0);
      
      #pragma omp parallel for private(i,j) default(shared)
      for(i = 0; i < nitem; i++){
        for(j = 0; j < ndim; j++){
          old_item_dist(i) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
          new_item_dist(i) += std::pow(old_z(k,j)-new_w(i,j), 2.0);
        }
        old_item_dist(i) = std::sqrt(old_item_dist(i));
        new_item_dist(i) = std::sqrt(new_item_dist(i));
        if(data(k,i) == 1.0){
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
      for(j = 0; j < ndim; j++){
        num += R::dnorm4(new_z(k,j), pr_mean_z, pr_sd_z, 1);
        den += R::dnorm4(old_z(k,j), pr_mean_z, pr_sd_z, 1);
      }
      ratio = update_like_item + (num - den);
      
      if(ratio > 0.0) accept = 1;
      else{
        un = R::runif(0,1);
        if(std::log(un) < ratio) accept = 1;
        else accept = 0;
      }
      
      if(accept == 1){
        for(j = 0; j < ndim; j++) old_z(k,j) = new_z(k,j);
        accept_z(k) += 1.0 / niter;
      }
      else{
        for(j = 0; j < ndim; j++) new_z(k,j) = old_z(k,j);
      }
    }
    
    for(i = 0; i < nitem; i++){
      new_beta(i) = R::rnorm(old_beta(i), jump_beta);
      old_like_beta = new_like_beta = 0.0;
      old_sample_dist.fill(0.0);
      
      #pragma omp parallel for private(k,j) default(shared)
      for(k = 0; k < nsample; k++){
        for(j = 0; j < ndim; j++) old_sample_dist(k) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
        old_sample_dist(k) = std::sqrt(old_sample_dist(k));
        if(data(k,i) == 1.0){
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
        accept_beta(i) += 1.0 / niter;
      }
      else{
        new_beta(i) = old_beta(i);
      }
    }
    
    for(k = 0; k < nsample; k++){
      new_theta(k) = R::rnorm(old_theta(k), jump_theta);
      old_like_theta = new_like_theta = 0.0;
      old_item_dist.fill(0.0);
      
      #pragma omp parallel for private(i,j) default(shared)
      for(i = 0; i < nitem; i++){
        for(j = 0; j < ndim; j++) old_item_dist(i) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
        old_item_dist(i) = std::sqrt(old_item_dist(i));
        if(data(k,i) == 1.0){
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
        accept_theta(k) += 1.0 / niter;
      }
      else new_theta(k) = old_theta(k);
    }
    
    if(iter > nburn && iter % nthin == 0){
      mle = 0.0; distance.fill(0.0);
      #pragma omp parallel for private(i,j,k) default(shared)
      for(k = 0; k < nsample; k++){
        for(i = 0; i < nitem; i++){
          for(j = 0; j < ndim; j++) distance(k,i) += std::pow(old_z(k,j)-old_w(i,j), 2.0);
          distance(k,i) = std::sqrt(distance(k,i));
          if(data(k,i) == 1.0) mle += -std::log(1.0 + std::exp(-(old_beta(i) + old_theta(k) - distance(k,i))));
          else mle += -std::log(1.0 + std::exp(old_beta(i) + old_theta(k) - distance(k,i)));
        }
      }
      
      for(k = 0; k < nsample; k++) mle += R::dnorm4(old_theta(k), pr_mean_theta, pr_sd_theta, 1);
      for(i = 0; i < nitem; i++) mle += R::dnorm4(old_beta(i), pr_mean_beta, pr_sd_beta, 1);
      for(k = 0; k < nsample; k++){
        for(j = 0; j < ndim; j++) mle += R::dnorm4(old_z(k,j), pr_mean_z, pr_sd_z, 1);
      }
      for(i = 0; i < nitem; i++){
        for(j = 0; j < ndim; j++) mle += R::dnorm4(old_w(i,j), pr_mean_w, pr_sd_w, 1);
      }
      
      for(k = 0; k < nsample; k++)
        for(j = 0; j < ndim; j++) sample_z(count,k,j) = old_z(k,j);
      for(i = 0; i < nitem; i++)
        for(j = 0; j < ndim; j++) sample_w(count,i,j) = old_w(i,j);
      for(i = 0; i < nitem; i++) sample_beta(count,i) = old_beta(i);
      for(k = 0; k < nsample; k++) sample_theta(count,k) = old_theta(k);
      sample_mle(count) = mle;
      count++;
    }
    
    if(iter % nprint == 0){
      printf("%.5d\n", iter);
      for(i = 0; i < nitem; i++) printf("% .3f  ", old_beta(i));  printf("\n");
      for(k = 0; k < nitem; k++) printf("% .3f  ", old_theta(k)); printf("\n");
      printf("%.3f %.3f %.3f %.3f\n", old_z(0,0), old_z(0,1), old_w(0,0), old_w(0,1));
    }
  }
  
  Rcpp::List output;
  output["beta"] = sample_beta;
  output["theta"] = sample_theta;
  output["z"] = sample_z;
  output["w"] = sample_w;
  
  output["accept_beta"] = accept_beta;
  output["accept_theta"] = accept_theta;
  output["accept_w"] = accept_w;
  output["accept_z"] = accept_z;
  output["posterior"] = mle;
  
  return(output);
}