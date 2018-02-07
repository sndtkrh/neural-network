#ifndef MATRIXLIB
#define MATRIXLIB
#include <iostream>
#include <iomanip>
#include "common.hpp"
#include "activation_functions.hpp"

vec mat_prod_vec(const mat & M, const vec & v){
  vec r(M.size(), 0);
  for(int i = 0; i < M.size(); i++){
    for(int j = 0; j < M[0].size(); j++){
      if( M[i].size() != v.size() ){
        throw "ERR";
      }
      r[i] += M[i][j] * v[j];
    }
  }
  return r;
}

vec vec_plus_vec(const vec & v, const vec & u){
  vec r(v.size());
  for(int i = 0; i < v.size(); i++){
    r[i] = v[i] + u[i];
  }
  return r;
}

vec function_apply_to_vec(ActivationFunction * af, const vec & v){
  vec r(v.size());
  for(int i = 0; i < v.size(); i++){
    r[i] = af->f(v[i]);
  }
  return r;
}

void print_mat(const mat & M){
  std::cout << std::fixed;
  std::cout << std::setprecision(2);
  for(int i = 0; i < M.size(); i++){
    for(int j = 0; j < M[i].size(); j++){
      std::cout << M[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

void print_vec(const vec & v){
  std::cout << std::fixed;
  std::cout << std::setprecision(2);
  for(int i = 0; i < v.size(); i++){
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}
#endif
