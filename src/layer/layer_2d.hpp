#ifndef LAYER2D
#define LAYER2D
#include "layer_base.hpp"

class Layer2D : public Layer {
public: 
  virtual void propagate() = 0;
  virtual void back_propagate() = 0;
  virtual void gradient_descent(F learning_rate, F momentum) = 0;
  
  int channel, unit_h, unit_w;
  int prev_channel, prev_h, prev_w;

  virtual void print_info( ){
    std::cout << layer_name << std::endl;
    std::cout << "  inputs = [ channel=" << prev_channel << ", h=" << prev_h << ", w=" << prev_w << "]" << std::endl;
    std::cout << "  units = [ channel=" << channel << ", h=" << unit_h << ", w=" << unit_w << "]" << std::endl;
    std::cout << "  activation function = " << activation_func->func_name << std::endl;
    std::cout << std::endl;
  }  
protected:
  int unit_coord( int c, int h, int w ){
    return c * unit_h * unit_w + h * unit_w + w;
  }
  int prev_coord( int c, int h, int w ){
    return c * prev_h * prev_w + h * prev_w + w;
  }
  bool is_in_prev(int c, int h, int w){
    return (0 <= c && c < prev_channel
            && 0 <= h && h < prev_h
            && 0 <= w && w < prev_w );
  }
  bool is_in_unit(int c, int h, int w){
    return (0 <= c && c < channel
            && 0 <= h && h < unit_h
            && 0 <= w && w < unit_w );
  }
};

#endif
