#define RCPP_ARMADILLO_FIX_Field
#include <RcppArmadillo.h>
#include <RcppParallel.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace RcppParallel;



// [[Rcpp::export]]
arma::field<arma::cube> Update_Field (arma::field<arma::cube> Field, arma::cube New_Slice, int n) {
  Field(n) = New_Slice;
  
  return Field;
}

// [[Rcpp::export]]
arma::field<arma::cube> Create_Field (int N, int d1, int d2, int d3) {
  arma::field<arma::cube> Field(N);
  arma::cube New_Slice(d1, d2, d3);
  Field.fill(New_Slice);
  
  return Field;
}



// [[Rcpp::export]]
bool dynSB_CheckConvergence (double J, double J_old, arma::colvec piv, arma::mat Pi, arma::mat B, arma::colvec piv_old, arma::mat Pi_old, arma::mat B_old, int it, double tol_lk, double tol_theta, int maxit) {
  int piv_dim = piv.n_elem;
  int Pi_dim = Pi.n_elem;
  int B_dim = B.n_elem;
  int theta_dim = piv_dim + Pi_dim + B_dim;
  
  arma::colvec Pi_vec(Pi.memptr(), Pi_dim, false);
  arma::colvec B_vec(B.memptr(), B_dim, false);
  arma::colvec theta(theta_dim);
  theta.subvec(0, piv_dim-1) = piv;
  theta.subvec(piv_dim, piv_dim+Pi_dim-1) = Pi_vec;
  theta.subvec(piv_dim+Pi_dim, piv_dim+Pi_dim+B_dim-1) = B_vec;
  
  arma::colvec Pi_old_vec(Pi_old.memptr(), Pi_dim, false);
  arma::colvec B_old_vec(B_old.memptr(), B_dim, false);
  arma::colvec theta_old(theta_dim);
  theta_old.subvec(0, piv_dim-1) = piv_old;
  theta_old.subvec(piv_dim, piv_dim+Pi_dim-1) = Pi_old_vec;
  theta_old.subvec(piv_dim+Pi_dim, piv_dim+Pi_dim+B_dim-1) = B_old_vec;
  
  bool J_conv = (abs(J - J_old)/abs(J_old) < tol_lk);
  bool theta_conv = (arma::max(arma::abs(theta - theta_old)) < tol_theta);
  bool maxit_reached = (it > maxit-1);
  bool minit_done = (it > 2);
  
  bool alt = (maxit_reached + (theta_conv && J_conv)) && minit_done;
  
  return alt;
}



// [[Rcpp::export]]
double dynSB_compute_ELBO (arma::cube Y, arma::colvec piv, arma::mat Pi, arma::mat B, arma::mat Tau1, arma::field<arma::cube> TAU, arma::cube Taum, int n, int k, int TT) {
  arma::mat Tau1_Clamped = Tau1;
  Tau1_Clamped.clamp(1e-12, 1-1e-12);
  arma::colvec piv_Clamped = piv;
  piv_Clamped.clamp(1e-12, 1-1e-12);
  arma::mat Pi_Clamped = Pi;
  Pi_Clamped.clamp(1e-12, 1-1e-12);
  
  double J = arma::accu(Tau1 * log(piv_Clamped)) - arma::accu(Tau1 % log(Tau1_Clamped));
  
  for (int i = 0; i < n; i++) {
    for (int t = 1; t < TT; t++) {
      arma::cube TAU_sub1 = TAU(t);
      arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
      arma::mat TAU_Clamped = TAU_sub2;
      TAU_Clamped.clamp(1e-12, 1-1e-12);
      arma::rowvec Taum_sub1 = Taum(arma::span(i), arma::span(), arma::span(t-1));
      arma::colvec Taum_sub2 = arma::conv_to<arma::colvec>::from(Taum_sub1);
      arma::rowvec Taum_sub3(k, arma::fill::ones);
      arma::mat Taum_sub4 = Taum_sub2 * Taum_sub3;
      J += arma::accu(Taum_sub4 % TAU_sub2 % (log(Pi_Clamped) - log(TAU_Clamped)));
    }
  }
  
  for (int i = 0; i < n-1; i++) {
    for (int t = 0; t < TT; t++) {
      arma::rowvec Y_sub1 = Y(arma::span(i), arma::span(i+1, n-1), arma::span(t));
      arma::colvec Y_sub2 = arma::conv_to<arma::colvec>::from(Y_sub1);
      for (int u = 0; u < k; u++) {
        for (int v = 0; v < k; v++) {
          double B_sub1 = B(u, v);
          if(B_sub1 < 1e-12) {
            B_sub1 = 1e-12;
          } else if(B_sub1 > 1-1e-12) {
            B_sub1 = 1-1e-12;
          }
          arma::colvec Taum_sub1 = Taum(arma::span(i+1, n-1), arma::span(v), arma::span(t));
          double Taum_sub2 = Taum(i, u, t);
          J += arma::accu(Taum_sub2 * Taum_sub1 % (Y_sub2 * log(B_sub1) + (1 - Y_sub2) * log(1 - B_sub1)));
        }
      }
    }
  }
  
  return J;
}




// [[Rcpp::export]]
Rcpp::List dynSB_compute_derivatives (arma::mat Tau1, arma::field<arma::cube> TAU, int n, int k, int TT) {
  // Taum
  arma::cube Taum(n, k, TT);
  Taum.slice(0) = Tau1;
  for(int i = 0; i < n; i++) {
    for(int t = 1; t < TT; t++) {
      arma::cube TAU_sub1 = TAU(t);
      arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
      arma::rowvec Taum_sub1 = Taum(arma::span(i), arma::span(), arma::span(t-1));
      arma::colvec Taum_sub2 = arma::conv_to<arma::colvec>::from(Taum_sub1);
      Taum(arma::span(i), arma::span(), arma::span(t)) = TAU_sub2.t() * Taum_sub2;
    }
  }
  
  // d1Taum
  arma::field<arma::mat> d1Taum(n, TT);
  for(int i = 0; i < n; i++) {
    d1Taum(i, 0) = arma::eye(k, k);
    for(int t = 1; t < TT; t++) {
      arma::cube TAU_sub1 = TAU(t);
      arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
      d1Taum(i, t) = TAU_sub2.t() * d1Taum(i, t-1);
    }
  }
  
  // d2Taum
  arma::field<arma::cube> d2Taum(n, TT, TT);
  arma::cube d2Taum_sub1(k, k, k, arma::fill::zeros);
  d2Taum.fill(d2Taum_sub1);
  for(int i = 0; i < n; i++) {
    for(int s = 1; s < TT; s++) {
      arma::cube d2Taum_sub2(k, k, k);
      for(int u = 0; u < k; u++) {
        d2Taum_sub2(arma::span(), arma::span(u), arma::span()) = Taum(i, u, s-1) * arma::eye(k, k);
      }
      d2Taum(i, s, s) = d2Taum_sub2;
      if(s < TT-1) {
        for(int t = s+1; t < TT; t++) {
          arma::cube TAU_sub1 = TAU(t);
          arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
          arma::cube d2Taum_sub3 = d2Taum(i, t, s);
          arma::cube d2Taum_sub4 = d2Taum(i, t-1, s);
          for(int u = 0; u < k; u++) {
            arma::mat d2Taum_sub5 = d2Taum_sub4(arma::span(), arma::span(u), arma::span());
            d2Taum_sub3(arma::span(), arma::span(u), arma::span()) = TAU_sub2.t() * d2Taum_sub5;
          }
          d2Taum(i, t, s) = d2Taum_sub3;
        }
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("Tau1") = Tau1,
                            Rcpp::Named("TAU") = TAU,
                            Rcpp::Named("Taum") = Taum,
                            Rcpp::Named("d1Taum") = d1Taum, 
                            Rcpp::Named("d2Taum") = d2Taum);
}



// [[Rcpp::export]]
Rcpp::List dynSB_Initial_Random (int n, int k, int TT) {
  // Tau1
  arma::mat Tau1_sub1(n, k, arma::fill::randu);
  arma::colvec Tau1_sub2 = 1/arma::sum(Tau1_sub1, 1);
  arma::rowvec Tau1_sub3(k, arma::fill::ones);
  arma::mat Tau1 = (Tau1_sub2 * Tau1_sub3) % Tau1_sub1;
  
  // TAU
  arma::field<arma::cube> TAU(TT);
  arma::cube TAU_sub1(n, k, k, arma::fill::zeros);
  TAU(0) = TAU_sub1;
  for(int t = 1; t < TT; t++) {
    arma::cube TAU_sub2(n, k, k, arma::fill::randu);
    for(int i = 0; i < n; i++) {
      arma::mat TAU_sub3 = TAU_sub2(arma::span(i), arma::span(), arma::span());
      arma::colvec TAU_sub4 = 1/arma::sum(TAU_sub3, 1);
      arma::rowvec TAU_sub5(k, arma::fill::ones);
      TAU_sub2(arma::span(i), arma::span(), arma::span()) = (TAU_sub4 * TAU_sub5) % TAU_sub3;
    }
    TAU(t) = TAU_sub2;
  }
  
  Rcpp::List out = dynSB_compute_derivatives(Tau1, TAU, n, k, TT);
  
  return out;
}



// [[Rcpp::export]]
Rcpp::List dynSB_Initial_Deterministic (arma::cube Y, int n, int k, int TT) {
  arma::mat YY(n*TT, n);
  for(int t = 0; t < TT; t++) {
    arma::mat Y_sub1 = Y(arma::span(), arma::span(), arma::span(t));
    Y_sub1.diag().zeros();
    YY(arma::span(t*n, t*n+n-1), arma::span()) = Y_sub1;
  }
  
  // Obtain environment containing function
  Rcpp::Environment base("package:stats");
  // Make function callable from C++
  Rcpp::Function kmeans_R = base["kmeans"];
  // Call the function
  Rcpp::List CL_sub1 = kmeans_R(Rcpp::_["x"] = YY, Rcpp::_["centers"] = k, Rcpp::_["nstart"] = 100);
  arma::colvec CL_sub2 = CL_sub1["cluster"];
  arma::mat CL = arma::reshape(CL_sub2, n, TT) - 1;
  
  // Tau1
  arma::mat Tau1(n, k, arma::fill::zeros);
  for(int i = 0; i < n; i++) {
    int Tau1_ind1 = CL(i, 0);
    Tau1(arma::span(i), arma::span(Tau1_ind1)) = 1;
  }
  
  // TAU
  arma::field<arma::cube> TAU(TT);
  arma::cube TAU_sub1(n, k, k, arma::fill::zeros);
  TAU(0) = TAU_sub1;
  for(int t = 1; t < TT; t++) {
    arma::cube TAU_sub2(n, k, k, arma::fill::zeros);
    for(int i = 0; i < n; i++) {
      int TAU_ind1 = CL(i, t);
      arma::cube TAU_sub3(1, k, 1, arma::fill::ones);
      TAU_sub2(arma::span(i), arma::span(), arma::span(TAU_ind1)) = TAU_sub3;
    }
    TAU(t) = TAU_sub2;
  }
  
  Rcpp::List out = dynSB_compute_derivatives(Tau1, TAU, n, k, TT);
  
  return out;
}



// [[Rcpp::export]]
Rcpp::List dynSB_VE_step (arma::cube Y, arma::colvec piv, arma::mat Pi, arma::mat B, arma::mat Tau1, arma::field<arma::cube> TAU, arma::cube Taum, arma::field<arma::mat> d1Taum, arma::field<arma::cube> d2Taum, int n, int k, int TT) {
  arma::mat Tau1_New = Tau1;
  arma::field<arma::cube> TAU_New = TAU;
  arma::cube Taum_New = Taum;
  arma::field<arma::mat> d1Taum_New = d1Taum;
  arma::field<arma::cube> d2Taum_New = d2Taum;
  
  arma::colvec piv_Clamped = piv;
  piv_Clamped.clamp(1e-12, 1-1e-12);
  arma::mat Pi_Clamped = Pi;
  Pi_Clamped.clamp(1e-12, 1-1e-12);
  
  // Tau1
  for(int i = 0; i < n; i++) {
    arma::colvec ltau = log(piv_Clamped);
    for(int t = 1; t < TT; t++) {
      arma::mat d1Taum_sub1 = d1Taum(i, t-1);
      arma::cube TAU_sub1 = TAU(t);
      arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
      arma::mat TAU_sub3 = TAU_sub2;
      TAU_sub3.clamp(1e-12, 1-1e-12);
      ltau += d1Taum_sub1.t() * arma::sum(TAU_sub2 % (log(Pi_Clamped) - log(TAU_sub3)), 1);
    }
    for(int j = 0; j < n; j++) {
      if(j != i) {
        for(int t = 0; t < TT; t++) {
          double Y_sub1 = Y(i, j, t);
          arma::mat B_sub1 = B;
          B_sub1.clamp(1e-12, 1-1e-12);
          arma::mat lB = Y_sub1 * log(B_sub1) + (1 - Y_sub1) * log(1 - B_sub1);
          arma::mat d1Taum_sub1 = d1Taum(i, t);
          arma::rowvec Taum_sub1 = Taum_New(arma::span(j), arma::span(), arma::span(t));
          arma::colvec Taum_sub2 = arma::conv_to<arma::colvec>::from(Taum_sub1);
          ltau += d1Taum_sub1.t() * lB * Taum_sub2;
        }
      }
    }
    arma::colvec Tau1_New_sub1 = exp(ltau - max(ltau))/arma::accu(exp(ltau - max(ltau)));
    Tau1_New(arma::span(i), arma::span()) = arma::conv_to<arma::rowvec>::from(Tau1_New_sub1);
    
    Taum_New(arma::span(i), arma::span(), arma::span(0)) = arma::conv_to<arma::rowvec>::from(Tau1_New_sub1);
    for(int t = 1; t < TT; t++) {
      arma::cube TAU_sub1 = TAU(t);
      arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
      arma::rowvec Taum_sub1 = Taum_New(arma::span(i), arma::span(), arma::span(t-1));
      arma::colvec Taum_sub2 = arma::conv_to<arma::colvec>::from(Taum_sub1);
      Taum_New(arma::span(i), arma::span(), arma::span(t)) = TAU_sub2.t() * Taum_sub2;
    }
    
    for(int s = 1; s < TT; s++) {
      arma::cube d2Taum_sub2(k, k, k);
      for(int u = 0; u < k; u++) {
        d2Taum_sub2(arma::span(), arma::span(u), arma::span()) = Taum_New(i, u, s-1) * arma::eye(k, k);
      }
      d2Taum_New(i, s, s) = d2Taum_sub2;
      if(s < TT-1) {
        for(int t = s+1; t < TT; t++) {
          arma::cube TAU_sub1 = TAU_New(t);
          arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
          arma::cube d2Taum_sub3 = d2Taum_New(i, t, s);
          arma::cube d2Taum_sub4 = d2Taum_New(i, t-1, s);
          for(int u = 0; u < k; u++) {
            arma::mat d2Taum_sub5 = d2Taum_sub4(arma::span(), arma::span(u), arma::span());
            d2Taum_sub3(arma::span(), arma::span(u), arma::span()) = TAU_sub2.t() * d2Taum_sub5;
          }
          d2Taum_New(i, t, s) = d2Taum_sub3;
        }
      }
    }
  }
  
  
  // TAU
  for(int i = 0; i < n; i++) {
    for(int s = 1; s < TT; s++) {
      for(int u = 0; u < k; u++) {
        arma::rowvec ltau_sub1 = log(Pi_Clamped(arma::span(u), arma::span()));
        arma::colvec ltau = arma::conv_to<arma::colvec>::from(ltau_sub1);
        if(s < TT-1) {
          for(int t = s+1; t < TT; t++) {
            arma::cube d2Taum_sub1 = d2Taum_New(i, t-1, s);
            arma::mat d2Taum_sub2 = d2Taum_sub1(arma::span(), arma::span(u), arma::span());
            arma::cube TAU_sub1 = TAU_New(t);
            arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
            arma::mat TAU_sub2_Clamped = TAU_sub2;
            TAU_sub2_Clamped.clamp(1e-12, 1-1e-12);
            double Taum_sub1 = Taum_New(i, u, s-1);
            if(Taum_sub1 < 1e-12) {
              Taum_sub1 = 1e-12;
            }
            ltau += (d2Taum_sub2.t() * arma::sum(TAU_sub2 % (log(Pi_Clamped) - log(TAU_sub2_Clamped)), 1))/Taum_sub1;
          }
        }
        for(int j = 0; j < n; j++) {
          if(j != i) {
            for(int t = s; t < TT; t++) {
              double Y_sub1 = Y(i, j, t);
              arma::mat B_sub1 = B;
              B_sub1.clamp(1e-12, 1-1e-12);
              arma::mat lB = Y_sub1 * log(B_sub1) + (1 - Y_sub1) * log(1 - B_sub1);
              arma::cube d2Taum_sub1 = d2Taum_New(i, t, s);
              arma::mat d2Taum_sub2 = d2Taum_sub1(arma::span(), arma::span(u), arma::span());
              arma::rowvec Taum_sub1 = Taum_New(arma::span(j), arma::span(), arma::span(t));
              arma::colvec Taum_sub2 = arma::conv_to<arma::colvec>::from(Taum_sub1);
              double Taum_sub3 = Taum_New(i, u, s-1);
              if(Taum_sub3 < 1e-12) {
                Taum_sub3 = 1e-12;
              }
              ltau += (d2Taum_sub2.t() * lB * Taum_sub2)/Taum_sub3;
            }
          }
        }
        arma::colvec TAU_New_sub1 = exp(ltau - max(ltau))/arma::accu(exp(ltau - max(ltau)));
        arma::cube TAU_New_sub2 = TAU_New(s);
        TAU_New_sub2(arma::span(i), arma::span(u), arma::span()) = TAU_New_sub1;
        TAU_New(s) = TAU_New_sub2;
        
        for(int t = s; t < TT; t++) {
          arma::cube TAU_sub1 = TAU_New(t);
          arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
          arma::rowvec Taum_sub1 = Taum_New(arma::span(i), arma::span(), arma::span(t-1));
          arma::colvec Taum_sub2 = arma::conv_to<arma::colvec>::from(Taum_sub1);
          Taum_New(arma::span(i), arma::span(), arma::span(t)) = TAU_sub2.t() * Taum_sub2;
        }
        
        for(int t = s; t < TT; t++) {
          arma::cube TAU_sub1 = TAU_New(t);
          arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
          d1Taum_New(i, t) = TAU_sub2.t() * d1Taum_New(i, t-1);
        }
        
        for(int t = s; t < TT; t++) {
          arma::cube d2Taum_sub1(k, k, k);
          for(int u = 0; u < k; u++) {
            d2Taum_sub1(arma::span(), arma::span(u), arma::span()) = Taum_New(i, u, t-1) * arma::eye(k, k);
          }
          d2Taum_New(i, t, t) = d2Taum_sub1;
          if(t < TT-1) {
            for(int t1 = t+1; t1 < TT; t1++) {
              arma::cube TAU_sub1 = TAU_New(t1);
              arma::mat TAU_sub2 = TAU_sub1(arma::span(i), arma::span(), arma::span());
              arma::cube d2Taum_sub3 = d2Taum_New(i, t1, t);
              arma::cube d2Taum_sub4 = d2Taum_New(i, t1-1, t);
              for(int u = 0; u < k; u++) {
                arma::mat d2Taum_sub5 = d2Taum_sub4(arma::span(), arma::span(u), arma::span());
                d2Taum_sub3(arma::span(), arma::span(u), arma::span()) = TAU_sub2.t() * d2Taum_sub5;
              }
              d2Taum_New(i, t1, t) = d2Taum_sub3;
            }
          }
        }
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("Tau1") = Tau1_New, 
                            Rcpp::Named("TAU") = TAU_New, 
                            Rcpp::Named("Taum") = Taum_New, 
                            Rcpp::Named("d1Taum") = d1Taum_New, 
                            Rcpp::Named("d2Taum") = d2Taum_New);
}



// [[Rcpp::export]]
Rcpp::List dynSB_M_step (arma::cube Y, arma::mat Tau1, arma::field<arma::cube> TAU, arma::cube Taum, int n, int k, int TT) {
  // B
  arma::mat B(k, k);
  for(int u = 0; u < k; u++) {
    for(int v = 0; v < k; v++) {
      double num = 0.0, den = 0.0;
      for(int i = 0; i < n-1; i++) {
        for(int t = 0; t < TT; t++){
          double Taum_sub1 = Taum(i, u, t);
          double Taum_sub2 = Taum(i, v, t);
          arma::colvec Taum_sub3 = Taum(arma::span(i+1, n-1), arma::span(v), arma::span(t));
          arma::colvec Taum_sub4 = Taum(arma::span(i+1, n-1), arma::span(u), arma::span(t));
          arma::rowvec Y_sub1 = Y(arma::span(i), arma::span(i+1, n-1), arma::span(t));// [[Rcpp::export]]
          arma::colvec Y_sub2 = arma::conv_to<arma::colvec>::from(Y_sub1);
          arma::colvec Y_sub3 = Y(arma::span(i+1, n-1), arma::span(i), arma::span(t));
          num += Taum_sub1 * arma::accu(Taum_sub3 % Y_sub2) + Taum_sub2 * arma::accu(Taum_sub4 % Y_sub3);
          den += Taum_sub1 * arma::accu(Taum_sub3) + Taum_sub2 * arma::accu(Taum_sub4);
        }
      }
      if(den == 0){
        B(u, v) = 0;
      } else {
        B(u, v) = num/den;
      }
    }
  }
  
  // piv
  arma::rowvec piv_sub1 = arma::mean(Tau1);
  arma::colvec piv = arma::conv_to<arma::colvec>::from(piv_sub1);
  
  // Pi
  arma::mat Pi(k, k, arma::fill::zeros);
  for(int u = 0; u < k; u++) {
    for(int v = 0; v < k; v++) {
      for(int t = 1; t < TT; t++) {
        arma::colvec Taum_sub1 = Taum(arma::span(), arma::span(u), arma::span(t-1));
        arma::cube TAU_sub1 = TAU(t);
        arma::colvec TAU_sub2 = TAU_sub1(arma::span(), arma::span(u), arma::span(v));
        Pi(u, v) += arma::accu(Taum_sub1 % TAU_sub2);
      }
    }
    double Pi_sub1 = arma::accu(Pi(arma::span(u), arma::span()));
    if(Pi_sub1 == 0) {
      arma::rowvec Pi_sub2(k, arma::fill::zeros);
      Pi_sub2(u) = 1;
      Pi(arma::span(u), arma::span()) = Pi_sub2;
    } else {
      Pi(arma::span(u), arma::span()) /= Pi_sub1;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("piv") = piv, 
                            Rcpp::Named("Pi") = Pi, 
                            Rcpp::Named("B") = B);
}



// [[Rcpp::export]]
double dynSB_compute_LogLik (arma::cube Y, arma::colvec piv, arma::mat Pi, arma::mat B, arma::umat V, int n, int k, int TT) {
  arma::colvec piv_Clamped = piv;
  piv_Clamped.clamp(1e-12, 1-1e-12);
  arma::mat Pi_Clamped = Pi;
  Pi_Clamped.clamp(1e-12, 1-1e-12);
  
  double LogLik = 0.0;
  for(int i = 0; i < n; i++) {
    LogLik += log(piv_Clamped(V(i, 0)));
    for(int t = 1; t < TT; t++) {
      LogLik += log(Pi_Clamped(V(i, t-1), V(i,t)));
    }
  }
  for(int i = 0; i < n-1; i++) {
    for(int t = 0; t < TT; t++) {
      arma::rowvec Y_sub1 = Y(arma::span(i), arma::span(i+1, n-1), arma::span(t));
      int V_sub1 = V(i, t);
      arma::uvec V_sub2(1); V_sub2(0) = V_sub1;
      arma::uvec V_sub3 = V(arma::span(i+1, n-1), arma::span(t));
      arma::rowvec B_sub1 = B(V_sub2, V_sub3);
      arma::rowvec B_sub2 = B_sub1;
      B_sub2.clamp(1e-12, 1-1e-12);
      LogLik += arma::accu(Y_sub1 % log(B_sub2) + (1 - Y_sub1) % log(1 - B_sub2));
    }
  }
  
  return LogLik;
}



// [[Rcpp::export]]
arma::umat dynSB_compute_Classification (arma::cube Taum, int n, int TT) {
  arma::umat V(n, TT);
  for(int t = 0; t < TT; t++) {
    arma::mat Taum_sub1 = Taum.slice(t);
    arma::uvec Taum_sub2 = arma::index_max(Taum_sub1, 1);
    V(arma::span(), arma::span(t)) = Taum_sub2;
  }
  
  return V;
}



// [[Rcpp::export]]
Rcpp::List dynSB_VEM (arma::cube Y, int k, Rcpp::List sv, double tol_lk, double tol_theta, int maxit) {
  double J;
  double J_old = R_NegInf;
  arma::vec J_vec(maxit + 1);
  
  int n = Y.n_cols;
  int TT = Y.n_slices;
  arma::colvec piv_old(k);
  arma::mat Pi_old(k, k); 
  arma::mat B_old(k, k);
  
  arma::mat Tau1_old = sv["Tau1"];
  arma::field<arma::cube> TAU_old = sv["TAU"];
  arma::cube Taum_old = sv["Taum"];
  arma::field<arma::mat> d1Taum_old = sv["d1Taum"];
  arma::field<arma::cube> d2Taum_old = sv["d2Taum"];
  
  int it = 0;
  bool alt = false;
  
  while(!alt) {
    it++;
    
    // 1. M-step
    Rcpp::List M = dynSB_M_step(Y, Tau1_old, TAU_old, Taum_old, n, k, TT);
    arma::colvec piv = M["piv"]; arma::mat Pi = M["Pi"]; arma::mat B = M["B"];
    
    // 2. VE-step
    Rcpp::List VE = dynSB_VE_step(Y, piv, Pi, B, Tau1_old, TAU_old, Taum_old, d1Taum_old, d2Taum_old, n, k, TT);
    arma::mat Tau1 = VE["Tau1"];
    arma::field<arma::cube> TAU = VE["TAU"];
    arma::cube Taum = VE["Taum"];
    arma::field<arma::mat> d1Taum = VE["d1Taum"];
    arma::field<arma::cube> d2Taum = VE["d2Taum"];
    
    // 3. Compute ELBO
    J = dynSB_compute_ELBO(Y, piv, Pi, B, Tau1, TAU, Taum, n, k, TT);
    J_vec(it-1) = J;
    
    // 4. Update convergence conditions
    if(it > 1) {
      alt = dynSB_CheckConvergence (J, J_old, piv, Pi, B, piv_old, Pi_old, B_old, it, tol_lk, tol_theta, maxit);
    }
    
    // 5. Update parameters
    J_old = J;
    piv_old = piv; Pi_old = Pi; B_old = B;
    Tau1_old = Tau1; TAU_old = TAU; Taum_old = Taum; d1Taum_old = d1Taum; d2Taum_old = d2Taum;
  }
  
  // 6. Additional M-step
  Rcpp::List M = dynSB_M_step(Y, Tau1_old, TAU_old, Taum_old, n, k, TT);
  arma::colvec piv = M["piv"]; arma::mat Pi = M["Pi"]; arma::mat B = M["B"];
  J = dynSB_compute_ELBO(Y, piv, Pi, B, Tau1_old, TAU_old, Taum_old, n, k, TT);
  J_vec(it) = J;
  
  // 7. Final LogLikelihood and Classification
  arma::umat V = dynSB_compute_Classification (Taum_old, n, TT);
  double LogLik = dynSB_compute_LogLik (Y, piv, Pi, B, V, n, k, TT);
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik, 
                            Rcpp::Named("J") = J, 
                            Rcpp::Named("J_vec") = J_vec.head(it), 
                            Rcpp::Named("it") = it,
                            Rcpp::Named("piv") = piv, 
                            Rcpp::Named("Pi") = Pi, 
                            Rcpp::Named("B") = B, 
                            Rcpp::Named("k") = k, 
                            Rcpp::Named("V") = V + 1,
                            Rcpp::Named("Tau1") = Tau1_old,
                            Rcpp::Named("TAU") = TAU_old);
}



// [[Rcpp::export]]
Rcpp::List dynSB_Update_step (arma::cube Y, int k, Rcpp::List sv, double tol_lk, double tol_theta, int maxit) {
  double J;
  double J_old = R_NegInf;
  
  int n = Y.n_cols;
  int TT = Y.n_slices;
  arma::colvec piv_old(k);
  arma::mat Pi_old(k, k); 
  arma::mat B_old(k, k);
  
  arma::mat Tau1_old = sv["Tau1"];
  arma::field<arma::cube> TAU_old = sv["TAU"];
  arma::cube Taum_old = sv["Taum"];
  arma::field<arma::mat> d1Taum_old = sv["d1Taum"];
  arma::field<arma::cube> d2Taum_old = sv["d2Taum"];
  
  int it = 0;
  bool alt = false;
  
  while(!alt) {
    it++;
    
    // 1. M-step
    Rcpp::List M = dynSB_M_step(Y, Tau1_old, TAU_old, Taum_old, n, k, TT);
    arma::colvec piv = M["piv"]; arma::mat Pi = M["Pi"]; arma::mat B = M["B"];
    
    // 2. VE-step
    Rcpp::List VE = dynSB_VE_step(Y, piv, Pi, B, Tau1_old, TAU_old, Taum_old, d1Taum_old, d2Taum_old, n, k, TT);
    arma::mat Tau1 = VE["Tau1"];
    arma::field<arma::cube> TAU = VE["TAU"];
    arma::cube Taum = VE["Taum"];
    arma::field<arma::mat> d1Taum = VE["d1Taum"];
    arma::field<arma::cube> d2Taum = VE["d2Taum"];
    
    // 3. Compute ELBO
    J = dynSB_compute_ELBO(Y, piv, Pi, B, Tau1, TAU, Taum, n, k, TT);
    
    // 4. Update convergence conditions
    if(it > 1) {
      alt = dynSB_CheckConvergence (J, J_old, piv, Pi, B, piv_old, Pi_old, B_old, it, tol_lk, tol_theta, maxit);
    }
    
    // 5. Update parameters
    J_old = J;
    piv_old = piv; Pi_old = Pi; B_old = B;
    Tau1_old = Tau1; TAU_old = TAU; Taum_old = Taum; d1Taum_old = d1Taum; d2Taum_old = d2Taum;
  }
  
  // 6. Additional M-step
  Rcpp::List M = dynSB_M_step(Y, Tau1_old, TAU_old, Taum_old, n, k, TT);
  arma::colvec piv = M["piv"]; arma::mat Pi = M["Pi"]; arma::mat B = M["B"];
  J = dynSB_compute_ELBO(Y, piv, Pi, B, Tau1_old, TAU_old, Taum_old, n, k, TT);
  
  // 7. Final LogLikelihood and Classification
  arma::umat V = dynSB_compute_Classification (Taum_old, n, TT);
  double LogLik = dynSB_compute_LogLik (Y, piv, Pi, B, V, n, k, TT);
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik, 
                            Rcpp::Named("J") = J, 
                            Rcpp::Named("piv") = piv, 
                            Rcpp::Named("Pi") = Pi, 
                            Rcpp::Named("Tau1") = Tau1_old,
                            Rcpp::Named("TAU") = TAU_old, 
                            Rcpp::Named("Taum") = Taum_old, 
                            Rcpp::Named("d1Taum") = d1Taum_old, 
                            Rcpp::Named("d2Taum") = d2Taum_old);
}



// [[Rcpp::export]]
Rcpp::List dynSB_LastUpdate_step (arma::cube Y, int k, Rcpp::List sv, double tol_lk, double tol_theta, int maxit) {
  double J;
  double J_old = R_NegInf;
  
  int n = Y.n_cols;
  int TT = Y.n_slices;
  arma::colvec piv_old(k);
  arma::mat Pi_old(k, k); 
  arma::mat B_old(k, k);
  
  arma::mat Tau1_old = sv["Tau1"];
  arma::field<arma::cube> TAU_old = sv["TAU"];
  arma::cube Taum_old = sv["Taum"];
  arma::field<arma::mat> d1Taum_old = sv["d1Taum"];
  arma::field<arma::cube> d2Taum_old = sv["d2Taum"];
  
  int it = 0;
  bool alt = false;
  
  while(!alt) {
    it++;
    
    // 1. M-step
    Rcpp::List M = dynSB_M_step(Y, Tau1_old, TAU_old, Taum_old, n, k, TT);
    arma::colvec piv = M["piv"]; arma::mat Pi = M["Pi"]; arma::mat B = M["B"];
    
    // 2. VE-step
    Rcpp::List VE = dynSB_VE_step(Y, piv, Pi, B, Tau1_old, TAU_old, Taum_old, d1Taum_old, d2Taum_old, n, k, TT);
    arma::mat Tau1 = VE["Tau1"];
    arma::field<arma::cube> TAU = VE["TAU"];
    arma::cube Taum = VE["Taum"];
    arma::field<arma::mat> d1Taum = VE["d1Taum"];
    arma::field<arma::cube> d2Taum = VE["d2Taum"];
    
    // 3. Compute ELBO
    J = dynSB_compute_ELBO(Y, piv, Pi, B, Tau1, TAU, Taum, n, k, TT);
    
    // 4. Update convergence conditions
    if(it > 1) {
      alt = dynSB_CheckConvergence (J, J_old, piv, Pi, B, piv_old, Pi_old, B_old, it, tol_lk, tol_theta, maxit);
    }
    
    // 5. Update parameters
    J_old = J;
    piv_old = piv; Pi_old = Pi; B_old = B;
    Tau1_old = Tau1; TAU_old = TAU; Taum_old = Taum; d1Taum_old = d1Taum; d2Taum_old = d2Taum;
  }
  
  // 6. Additional M-step
  Rcpp::List M = dynSB_M_step(Y, Tau1_old, TAU_old, Taum_old, n, k, TT);
  arma::colvec piv = M["piv"]; arma::mat Pi = M["Pi"]; arma::mat B = M["B"];
  J = dynSB_compute_ELBO(Y, piv, Pi, B, Tau1_old, TAU_old, Taum_old, n, k, TT);
  
  // 7. Final LogLikelihood and Classification
  arma::umat V = dynSB_compute_Classification (Taum_old, n, TT);
  double LogLik = dynSB_compute_LogLik (Y, piv, Pi, B, V, n, k, TT);
  
  
  return Rcpp::List::create(Rcpp::Named("LogLik") = LogLik, 
                            Rcpp::Named("J") = J, 
                            Rcpp::Named("it") = it,
                            Rcpp::Named("piv") = piv, 
                            Rcpp::Named("Pi") = Pi, 
                            Rcpp::Named("B") = B, 
                            Rcpp::Named("k") = k, 
                            Rcpp::Named("V") = V + 1,
                            Rcpp::Named("Tau1") = Tau1_old,
                            Rcpp::Named("TAU") = TAU_old);
}



// [[Rcpp::export]]
Rcpp::List dynSB_Mutation_step (arma::mat Tau1, arma::field<arma::cube> TAU, int n, int k, int TT, double LMP, double UMP) {
  double prob_mut;
  if(LMP == UMP) {
    prob_mut = LMP;
  } else {
    prob_mut = arma::randu(arma::distr_param(LMP, UMP));
  }
  
  // Tau1
  for(int i = 0; i < n; i++) {
    arma::rowvec Old_sub1 = Tau1(arma::span(i), arma::span());
    int Old = arma::index_max(Old_sub1);
    int New = arma::randi(arma::distr_param(0, k-2));
    if(New >= Old) {
      New += 1;
    }
    double RandUnif = arma::randu();
    if (RandUnif <= prob_mut) {
      double aux = Tau1(i, Old);
      Tau1(i, Old) = Tau1(i, New);
      Tau1(i, New) = aux;
    }
  }
  
  // TAU
  for(int t = 1; t < TT; t++) {
    arma::cube TAU_sub1 = TAU(t);
    for(int i = 0; i < n; i++){
      for(int u = 0; u < k; u++) {
        arma::rowvec Old_sub1 = TAU_sub1(arma::span(i), arma::span(u), arma::span());
        int Old = arma::index_max(Old_sub1);
        int New = arma::randi(arma::distr_param(0, k-2));
        if(New >= Old) {
          New += 1;
        }
        double RandUnif = arma::randu();
        if (RandUnif <= prob_mut) {
          double aux = TAU_sub1(i, u, Old);
          TAU_sub1(i, u, Old) = TAU_sub1(i, u, New);
          TAU_sub1(i, u, New) = aux;
        }
      }
    }
    TAU(t) = TAU_sub1;
  }
  
  // Taum, d1Taum, d2Taum
  Rcpp::List out = dynSB_compute_derivatives(Tau1, TAU, n, k, TT);
  
  return out;
}



// [[Rcpp::export]]
Rcpp::List dynSB_TotalSelection_step (Rcpp::List P_a, Rcpp::List P_b, Rcpp::List P_c, arma::rowvec fit_a, arma::rowvec fit_b, arma::rowvec fit_c) {
  int n_parents = fit_a.n_cols;
  Rcpp::List PV(n_parents);
  Rcpp::List PV_Aux(3*n_parents);
  arma::rowvec fit_Aux(3*n_parents);
  
  for (int b = 0; b < n_parents; b++) {
    PV_Aux(b) = P_a(b);
    fit_Aux(b) = fit_a(b);
  }
  for (int b = 0; b < n_parents; b++) {
    PV_Aux(n_parents + b) = P_b(b);
    fit_Aux(n_parents + b) = fit_b(b);
  }
  for (int b = 0; b < n_parents; b++) {
    PV_Aux(2*n_parents + b) = P_c(b);
    fit_Aux(2*n_parents + b) = fit_c(b);
  }
  fit_Aux.replace(arma::datum::nan, R_NegInf);
  
  arma::uvec indeces = arma::sort_index(fit_Aux, "descend");
  
  for (int b = 0; b < n_parents; b++) {
    int ind = indeces(b);
    PV(b) = PV_Aux(ind);
  }
  
  return PV;
}



// [[Rcpp::export]]
Rcpp::List dynSB_Selection_step (Rcpp::List P_Old, Rcpp::List P_New, arma::rowvec fit_Old, arma::rowvec fit_New, bool keep_best) {
  int n_parents = fit_New.n_cols;
  Rcpp::List PV(n_parents);
  
  arma::rowvec fit_Old_1 = fit_Old.head(1);
  bool best_isnan = fit_Old_1.has_nan();
  int b_first = 0;
  if (keep_best && !best_isnan) {
    PV(0) = P_Old(0);
    b_first = 1;
  }
  
  Rcpp::List PV_Aux(2*n_parents - b_first);
  arma::rowvec fit_Aux(2*n_parents - b_first);
  
  for (int b = b_first; b < n_parents; b++) {
    PV_Aux(b - b_first) = P_Old(b);
    fit_Aux(b - b_first) = fit_Old(b);
  }
  for (int b = 0; b < n_parents; b++) {
    PV_Aux(n_parents + b - b_first) = P_New(b);
    fit_Aux(n_parents + b - b_first) = fit_New(b);
  }
  fit_Aux.replace(arma::datum::nan, R_NegInf);
  
  arma::uvec indeces = arma::sort_index(fit_Aux, "descend");
  
  for (int b = b_first; b < n_parents; b++) {
    int ind = indeces(b - b_first);
    PV(b) = PV_Aux(ind);
  }
  
  return PV;
}



// [[Rcpp::export]]
Rcpp::List dynSB_Random_Initialization_step (arma::cube Y, int n, int k, int TT, int n_parents) {
  Rcpp::List P(n_parents);
  
  Function Init_Random_Cpp("Init_Random");
  for(int b = 0; b < n_parents; b++) {
    Rcpp::List svR = Init_Random_Cpp(n, k, TT);
    P(b) = svR;
  }
  
  return P;
}



// [[Rcpp::export]]
Rcpp::List dynSB_Kmeans_Initialization_step (arma::cube Y, int n, int k, int TT, int n_parents, double LMP, double UMP) {
  Rcpp::List P(n_parents);
  
  Function Init_Kmeans_Cpp("Init_Kmeans");
  Rcpp::List svD = Init_Kmeans_Cpp(Y, n, k, TT);
  P(0) = svD;
  arma::mat svD_Tau1 = svD["Tau1"];
  arma::field<arma::cube> svD_TAU = svD["TAU"];
  
  for(int b = 1; b < n_parents; b++) {
    Rcpp::List svD_mut = dynSB_Mutation_step(svD_Tau1, svD_TAU, n, k, TT, LMP, UMP);
    P(b) = svD_mut;
  }
  
  return P;
}



// [[Rcpp::export]]
Rcpp::List dynSB_SBM_Initialization_step (arma::cube Y, int n, int k, int TT, int n_parents, double LMP, double UMP) {
  Rcpp::List P(n_parents);
  
  Function Init_SBM_Weighted_Cpp("Init_SBM_Weighted");
  Rcpp::List svD = Init_SBM_Weighted_Cpp(Y, n, k, TT);
  P(0) = svD;
  arma::mat svD_Tau1 = svD["Tau1"];
  arma::field<arma::cube> svD_TAU = svD["TAU"];
  
  for(int b = 1; b < n_parents; b++) {
    Rcpp::List svD_mut = dynSB_Mutation_step(svD_Tau1, svD_TAU, n, k, TT, LMP, UMP);
    P(b) = svD_mut;
  }
  
  return P;
}



// [[Rcpp::export]]
Rcpp::List dynSB_EVEM_step (arma::cube Y, int k, Rcpp::List PV1, double tol_lk, double tol_theta, int n_parents, double LMP, double UMP, int R, int n, int TT, bool keep_best) {
  Rcpp::List PV2(n_parents), PV3(n_parents), PV4(n_parents), PV5(n_parents);
  arma::rowvec fit2(n_parents), fit4(n_parents), fit5(n_parents);
  double fit_old;
  int b_first = 0;
  
  int it = 0;
  bool alt = false;
  
  while (!alt) {
    it ++;
    
    // 1. Update parents and compute fitness
    for (int b = 0; b < n_parents; b++) {
      Rcpp::List PV1_sub1 = PV1(b);
      Rcpp::List PV1_sub2 = dynSB_Update_step(Y, k, PV1_sub1, tol_lk, tol_theta, R);
      double PV2_J = PV1_sub2["J"];
      PV2(b) = PV1_sub2;
      fit2(b) = PV2_J;
    }
    // 2. Mutation
    if (keep_best) {
      PV3(0) = PV2(0);
      b_first = 1;
    }
    for (int b = b_first; b < n_parents; b++) {
      Rcpp::List PV2_sub1 = PV2(b);
      arma::mat Tau1 = PV2_sub1["Tau1"];
      arma::field<arma::cube> TAU = PV2_sub1["TAU"];
      Rcpp::List PV2_sub2 = dynSB_Mutation_step(Tau1, TAU, n, k, TT, LMP, UMP);
      PV3(b) = PV2_sub2;
    }
    // 3. Update children and compute fitness
    for (int b = 0; b < n_parents; b++) {
      Rcpp::List PV3_sub1 = PV3(b);
      Rcpp::List PV3_sub2 = dynSB_Update_step(Y, k, PV3_sub1, tol_lk, tol_theta, R);
      PV4(b) = PV3_sub2;
      fit4(b) = PV3_sub2["J"];
    }
    // 4. Select new parents
    PV5 = dynSB_Selection_step(PV2, PV4, fit2, fit4, keep_best);
    for(int b = 0; b < n_parents; b++) {
      Rcpp::List PV5_sub1 = PV5(b);
      fit5(b) = PV5_sub1["J"];
    }
    
    if (it > 1) {
      alt = (abs(fit5.max() - fit_old)/abs(fit_old) < tol_lk);
    }
    fit_old = fit5.max();
    PV1 = PV5;
  }
  
  return Rcpp::List::create(Rcpp::Named("PV") = PV5,
                            Rcpp::Named("J") = fit5);
}



// [[Rcpp::export]]
Rcpp::List dynSB_EVEM (arma::cube Y, int k, double tol_lk, double tol_theta, int maxit, int n_parents, double LMP, double UMP, int R) {
  int n = Y.n_cols;
  int TT = Y.n_slices;
  
  // 1. Initial values
  Rcpp::List PV0_Ran = dynSB_Random_Initialization_step(Y, n, k, TT, n_parents);
  Rcpp::List PV0_Det1 = dynSB_Kmeans_Initialization_step(Y, n, k, TT, n_parents, LMP, UMP);
  Rcpp::List PV0_Det2 = dynSB_SBM_Initialization_step(Y, n, k, TT, n_parents, LMP, UMP);
  
  // 2. Evolution (separate)
  Rcpp::List PV1_Ran_Aux = dynSB_EVEM_step(Y, k, PV0_Ran, tol_lk, tol_theta, n_parents, LMP, UMP, R, n, TT, false);
  arma::rowvec fit1_Ran = PV1_Ran_Aux["J"];
  Rcpp::List PV1_Ran = PV1_Ran_Aux["PV"];
  Rcout << "Evolution Random.\n";
  Rcpp::List PV1_Det1_Aux = dynSB_EVEM_step(Y, k, PV0_Det1, tol_lk, tol_theta, n_parents, LMP, UMP, R, n, TT, true);
  arma::rowvec fit1_Det1 = PV1_Det1_Aux["J"];
  Rcpp::List PV1_Det1 = PV1_Det1_Aux["PV"];
  Rcout << "Evolution Deterministic (K-means).\n";
  Rcpp::List PV1_Det2_Aux = dynSB_EVEM_step(Y, k, PV0_Det2, tol_lk, tol_theta, n_parents, LMP, UMP, R, n, TT, true);
  arma::rowvec fit1_Det2 = PV1_Det2_Aux["J"];
  Rcpp::List PV1_Det2 = PV1_Det2_Aux["PV"];
  Rcout << "Evolution Deterministic (SBM).\n";
  
  // 3. Select best
  Rcpp::List PV2 = dynSB_TotalSelection_step(PV1_Ran, PV1_Det1, PV1_Det2, fit1_Ran, fit1_Det1, fit1_Det2);
  Rcout << "Selection.\n";
  
  // 4. Evolution (in common)
  Rcpp::List PV3_Aux = dynSB_EVEM_step(Y, k, PV2, tol_lk, tol_theta, n_parents, LMP, UMP, R, n, TT, false);
  Rcpp::List PV3 = PV3_Aux["PV"];
  arma::rowvec fit3 = PV3_Aux["J"];
  Rcout << "Evolution Common.\n";
  
  // 5. Last EM step
  int ind_best = fit3.index_max();
  Rcpp::List PV_best = PV3(ind_best);
  Rcpp::List out_best = dynSB_LastUpdate_step(Y, k, PV_best, tol_lk, tol_theta, maxit);
  Rcout << "Last Step.\n";
  
  return out_best;
}
