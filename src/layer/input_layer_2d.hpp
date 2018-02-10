#ifndef INPUTLAYER2D
#define INPUTLAYER2D
#include "layer_2d.hpp"

class InputLayer2D : public Layer2D {
public:
  InputLayer2D( int ch, int h, int w ){
    channel = ch;
    unit_h = h;
    unit_w = w;
    units = channel * unit_h * unit_w;
    unit_output.resize( units, 0 );
    activated_output.resize( units, 0 );
    delta.resize( units, 0 );
    activation_func = &id;
    layer_name = "[input 2D]";
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
  vec input_vec;
private:
};

#endif
