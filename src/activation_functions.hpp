#ifndef ACTIVATIONFUNCTION
#define ACTIVATIONFUNCTION
#include <iostream>
#include "common.hpp"

class ActivationFunction{
public:
  std::string func_name;
  virtual F f(F u) = 0;
  virtual F df(F u) = 0;
};

class Id : public ActivationFunction {
public:
  Id(){
    func_name = "Id";
  }
  F f(F u){
    return u;
  }
  F df(F u){
    return 1.0;
  }
} id;

class ReLU : public ActivationFunction {
  // rectified linear function
public:
  ReLU(){
    func_name = "ReLU";
  }
  F f(F u){
    return std::max(F(0), u);
  }
  F df(F u){
    return (u < 0) ? 0 : 1.0;
  }
} relu;

class Sigmoid : public ActivationFunction {
public:
  Sigmoid(){
    func_name = "sigmoid";
  }
  F f(F u){
    return 1.0 / (1.0 + std::exp(-u));
  }
  F df(F u){
    return f(u) * (1 - f(u));
  }
} sigmoid;

class Softmax : public ActivationFunction {
  // this is a damy class
public:
  Softmax(){
    func_name = "softmax";
  }
  F f(F u){
    return 0;
  }
  F df(F u){
    return 0;
  }
} softmax;
#endif
