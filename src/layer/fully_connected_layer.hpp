#ifndef FULLLYCONNECTEDLAYER
#define FULLLYCONNECTEDLAYER
#include "layer_base.hpp"

class FullyConnectedLayer : public Layer {
public:
  FullyConnectedLayer(int u, Layer * prev, ActivationFunction * af, std::string ln){
    init( u, prev, af, "[fully connected]" + ln );
    weight.resize( units );
    dweight.resize( units );
    sum_square_grad_weight.resize( units );
    for(int i = 0; i < units; i++){
      weight[i].resize( inputs );
      dweight[i].resize( inputs );
      sum_square_grad_weight[i].resize( inputs, 0 );
    }
    bias.resize( units, 0 );
    dbias.resize( units, 0 );
    sum_square_grad_bias.resize( units, 0 );
    // random initialization
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
    if( next_layer == nullptr ){
      compute_this_layer_delta();
    }
    compute_previous_layer_delta();
    if( previous_layer != nullptr )
      previous_layer->back_propagate();
  }
  void compute_previous_layer_delta(){
    vec & prev_delta = previous_layer->delta;
    std::fill( prev_delta.begin(), prev_delta.end(), 0 );
    for(int pu = 0; pu < inputs; pu++){
      for(int u = 0; u < units; u++){
        prev_delta[pu] += delta[u] * weight[u][pu] * previous_layer->activation_func->df( previous_layer->unit_output[pu] );
      }
    }
  }
  void compute_this_layer_delta(){
    std::fill( delta.begin(), delta.end(), 0 );
    for(int u = 0; u < units; u++){
      delta[u] = activated_output[u] - target[u];
    }
  }
  virtual void gradient_descent( F learning_rate, F momentum ){
    vec & z = previous_layer->activated_output;
    for(int i = 0; i < units; i++){
      for(int j = 0; j < inputs; j++){
        F grad = delta[i] * z[j];
        // AdaGrad
        sum_square_grad_weight[i][j] += grad * grad;
        dweight[i][j] = - learning_rate * grad / std::sqrt( std::max(sum_square_grad_weight[i][j], (F)1.0) ) + momentum * dweight[i][j];
        weight[i][j] += dweight[i][j];
      }
    }
    for(int i = 0; i < units; i++){
      F grad = delta[i];
      // AdaGrad
      sum_square_grad_bias[i] += grad * grad;
      dbias[i] = - learning_rate * grad / std::sqrt( std::max(sum_square_grad_bias[i], (F)1.0) ) + momentum * dbias[i];
      bias[i] += dbias[i];
    }
    if( next_layer != nullptr ){
      next_layer->gradient_descent( learning_rate, momentum );
    }
  }
  void print_weight(){
    print_mat( weight );
  }
protected:
  mat weight;
  mat dweight;
  mat sum_square_grad_weight;
  vec bias;
  vec dbias;
  vec sum_square_grad_bias;
};

#endif
