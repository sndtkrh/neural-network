#ifndef LAYERBASE
#define LAYERBASE
#include <random>
#include <algorithm>
#include "../common.hpp"
#include "../activation_functions.hpp"
#include "../matrix.hpp"

class Layer{
public:
  Layer * previous_layer;
  Layer * next_layer;
  int units;
  int inputs;
  ActivationFunction * activation_func;
  vec unit_output, activated_output;
  vec delta;
  std::string layer_name;

  virtual void propagate() = 0;
  virtual void back_propagate() = 0;
  virtual void gradient_descent(F learning_rate, F momentum) = 0;

  void set_target( vec & t ){
    target = t;
  }

  virtual void print_info( ){
    std::cout << layer_name << std::endl;
    std::cout << "  inputs = " << inputs << std::endl;
    std::cout << "  units = " << units << std::endl;
    std::cout << "  activation function = " << activation_func->func_name << std::endl;
    std::cout << std::endl;
  }
protected:
  vec target;
  void init( int u, Layer * prev, ActivationFunction * af, std::string ln) {
    next_layer = nullptr;
    previous_layer = prev;
    previous_layer->next_layer = this;
    units = u;
    inputs = prev->units;
    activation_func = af;
    layer_name = ln;
    unit_output.resize( units );
    activated_output.resize( units );
    delta.resize( units, 0 );
  }
};

#endif
