#ifndef ACTIVATIONFUNCTION
#define ACTIVATIONFUNCTION
#include <iostream>
#include "common.hpp"

class ActivationFunction{
public:
  std::string func_name;
  ActivationFunction(){
    func_name = "id";
  }
  virtual F f(F u){
    return u;
  }
  virtual F df(F u){
    return 1;
  }
};

class ReLU : public ActivationFunction {
  // rectified linear function
public:
  ReLU(){
    func_name = "ReLU";
  }
  F f(F u){
    return std::max(F(0.0), u);
  }
  F df(F u){
    return (u < 0) ? 0 : 1;
  }
};

#endif
