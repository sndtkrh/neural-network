#ifndef SOFTMAXLAYER
#define SOFTMAXLAYER
#include "fully_connected_layer.hpp"

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

class SoftmaxLayer : public FullyConnectedLayer {
public:
  SoftmaxLayer(int u, Layer * prev ) : FullyConnectedLayer( u, prev, &softmax, "[softmax]" ){}
  
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
    // compute this layer's delta
    for(int i = 0; i < units; i++){
      // differenciate cross entropy
      delta[i] = activated_output[i] - target[i];
    }
    // compute previous layer's delta
    compute_previous_layer_delta();
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

#endif
