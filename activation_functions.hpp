#ifndef ACTIVATIONFUNCTION
#define ACTIVATIONFUNCTION
#include "common.hpp"

class ActivationFunction{
public:
  ActivationFunction(){
  }
  virtual F f(F u){
    return u;
  }
  virtual F df(F u){
    return 1;
  }
};

class ReL : public ActivationFunction {
  // rectified linear function
public:
  F f(F u){
    return std::max(F(0.0), u);
  }
  F df(F u){
    return (u < 0) ? 0 : 1;
  }
};

#endif
