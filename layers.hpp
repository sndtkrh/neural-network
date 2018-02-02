#ifndef LAYERS
#define LAYERS
#include <random>
#include <algorithm>
#include "common.hpp"
#include "activation_functions.hpp"
#include "matrix.hpp"

class Layer{
public:
  Layer * previous_layer;
  Layer * next_layer;
  int units;
  int inputs;
  ActivationFunction activation_func;
  vec unit_output, activated_output;
  vec delta;
  std::string layer_name;

  virtual void propagate() = 0;
  virtual void back_propagate() = 0;
  virtual void gradient_descent(F learning_rate, F momentum) = 0;
  virtual F get_weight(int next_unit, int this_unit) = 0;
  ActivationFunction func(){
    return activation_func;
  }
  std::string name(){
    return layer_name;
  }

  void set_target( vec & t ){
    target = t;
  }

protected:
  vec target;
  void init( int u, Layer * prev, ActivationFunction af, std::string ln) {
    next_layer = nullptr;
    previous_layer = prev;
    units = u;
    inputs = prev->units;
    activation_func = af;
    layer_name = ln;
    unit_output.resize( units );
    activated_output.resize( units );
    delta.resize( units );
  }
};


class FullyConnectedLayer : public Layer {
public:
  FullyConnectedLayer(int u, Layer * prev, ActivationFunction af, std::string ln){
    init( u, prev, af, "fully connected : " + ln );
    weight.resize( units );
    dweight.resize( units );
    sum_square_grad.resize( units );
    previous_layer->next_layer = this;
    for(int i = 0; i < units; i++){
      weight[i].resize( inputs );
      dweight[i].resize( inputs );
      sum_square_grad[i].resize( inputs, 0 );
    }
    bias.resize( units, 0 );
    dbias.resize( units, 0 );

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> dist(0.0, 0.1);
    for(int i = 0; i < units; i++){
      for(int j = 0; j < inputs; j++){
	weight[i][j] = dist(engine);
      }
    }
  }
  virtual void propagate(){
    vec & z = previous_layer->activated_output;
    unit_output = vec_plus_vec(mat_prod_vec( weight, z ), bias);
    activated_output = function_apply_to_vec(activation_func, unit_output);
    if( next_layer != nullptr )
      next_layer->propagate();
  }
  virtual void back_propagate() {
    vec & next_delta = next_layer->delta;
    std::fill( delta.begin(), delta.end(), 0 );
    for(int k = 0; k < next_layer->units; k++){
      for(int j = 0; j < units; j++){
	delta[j] += next_delta[k] * get_weight(k, j) * activation_func.df( unit_output[j] );
      }
    }
    if( previous_layer != nullptr )
      previous_layer->back_propagate();
  }
  virtual void gradient_descent( F learning_rate, F momentum ){
    vec & z = previous_layer->activated_output;
    for(int i = 0; i < units; i++){
      for(int j = 0; j < inputs; j++){
	F grad = delta[i] * z[j];
	// AdaGrad
	sum_square_grad[i][j] += grad * grad;
	dweight[i][j] = - learning_rate * grad / std::max( std::sqrt(sum_square_grad[i][j]), (F)1.0 ) + momentum * dweight[i][j];
	weight[i][j] += dweight[i][j];
      }
    }
    for(int i = 0; i < units; i++){
      dbias[i] = -learning_rate * delta[i] + momentum * dbias[i];
      bias[i] += dbias[i];
    }
    if( next_layer != nullptr )
      next_layer->gradient_descent( learning_rate, momentum );
  }
  F get_weight(int next_unit, int this_unit){
    return weight[next_unit][this_unit];
  }
protected:
  mat weight;
  mat dweight;
  vec bias;
  vec dbias;

  mat sum_square_grad;
};

class SoftmaxLayer : public FullyConnectedLayer {
public:
  SoftmaxLayer(int u, Layer * prev ) : FullyConnectedLayer( u, prev, ActivationFunction(), "softmax" ){}
  
  void propagate(){
    vec & z = previous_layer->activated_output;
    unit_output = vec_plus_vec(mat_prod_vec( weight, z ), bias);
    F sum = 0;
    for(int i = 0; i < units; i++){
      sum += std::exp( unit_output[i] );
    }
    for(int i = 0; i < units; i++){
      activated_output[i] = std::exp( unit_output[i] ) / sum;
    }
    if( next_layer != nullptr )
      next_layer->propagate();
  }
  void back_propagate(){
    for(int i = 0; i < units; i++){
      // differenciate cross entropy
      delta[i] = activated_output[i] - target[i];
    }
    if( previous_layer != nullptr )
      previous_layer->back_propagate();
  }
  int get_class(){
    F p = activated_output[0];
    int i = 0;
    for(int k = 1; k < units; k++){
      if( p < activated_output[k] ){
	p = activated_output[k];
	i = k;
      }
    }
    return i;
  }
};

class InputLayer : public Layer {
public:
  InputLayer( int u ){
    units = u;
    unit_output.resize( units );
    activated_output.resize( units );
    layer_name = "input";
  }
  void propagate( vec & in ) {
    input_vec = in;
    propagate();
  }
  void propagate(){
    unit_output = input_vec;
    activated_output = input_vec;
    next_layer->propagate();
  }
  void back_propagate(){
    return;
  }
  void gradient_descent(F learning_rate, F momentum){
    if( next_layer != nullptr )
      next_layer->gradient_descent( learning_rate, momentum );
  }
  F get_weight(int v, int u){
    return (v==u) ? 1.0 : 0;
  }
private:
  vec input_vec;
};

#endif
