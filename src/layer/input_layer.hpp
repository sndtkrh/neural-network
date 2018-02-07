#ifndef INPUTLAYER
#define INPUTLAYER
#include "layer_base.hpp"

class InputLayer : public Layer {
public:
  InputLayer( int u ){
    units = u;
    unit_output.resize( units );
    delta.resize( units );
    activated_output.resize( units );
    activation_func = &id;
    layer_name = "[input]";
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
  void print_network_info( ){
    Layer * l = this;
    while( l != nullptr ){
      l->print_info();
      l = l->next_layer;
    }
  }
private:
  vec input_vec;
};

#endif
