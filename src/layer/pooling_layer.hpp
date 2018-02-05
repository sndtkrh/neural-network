#ifndef POOLINGLAYER
#define POOLINGLAYER
#include "layer_base.hpp"
#include "layer_2d.hpp"

class MaxPoolingLayer : public Layer2D {
public:
  MaxPoolingLayer(){}
  MaxPoolingLayer(int ps, int st, Layer2D * prev, ActivationFunction af, std::string ln){
    pooling_size = ps;
    stride = st;
    channel = prev_channel = prev->channel;
    prev_h = prev->unit_h;
    prev_w = prev->unit_w;
    unit_h = prev_h / stride;
    unit_w = prev_w / stride;
    init( channel * unit_h * unit_w, prev, af, "pooling : " + ln );
    unit_max_coord.resize( units );
  }

  void propagate(){
    for(int c = 0; c < channel; c++){
      for(int h = 0; h < unit_h; h++){
	for(int w = 0; w < unit_w; w++){
	  int mph = -1, mpw = -1;
	  F mv = -inf;
	  for(int s = 0; s < pooling_size; s++){
	    for(int t = 0; t < pooling_size; t++){
	      int ph = h * stride + s - pooling_size / 2;
	      int pw = w * stride + t - pooling_size / 2;
	      if( is_in_prev( c, ph, pw ) && mv < previous_layer->activated_output[ prev_coord( c, ph, pw ) ] ){
	        mv = previous_layer->activated_output[ prev_coord( c, ph, pw ) ];
		mph = ph;
		mpw = pw;
	      }
	    }
	  }
	  int unit_idx = unit_coord(c, h, w);
	  unit_max_coord[ unit_idx ] = std::make_pair(mph, mpw);
	  unit_output[ unit_idx ] = mv;
	  activated_output[ unit_idx ] = activation_func.f( mv );
	}
      }
    }
    if( next_layer != nullptr )
      next_layer->propagate();
  }

  void back_propagate(){
    vec & prev_delta = previous_layer->delta;
    std::fill( prev_delta.begin(), prev_delta.end(), 0 );
    for(int c = 0; c < channel; c++){
      for(int h = 0; h < unit_h; h++){
	for(int w = 0; w < unit_w; w++){
	  int ph = unit_max_coord[ unit_coord(c, h, w) ].first;
	  int pw = unit_max_coord[ unit_coord(c, h, w) ].second;
	  prev_delta[ prev_coord(c, ph, pw) ]
	    += delta[ unit_coord(c, h, w) ] * previous_layer->activation_func.df( previous_layer->unit_output[ prev_coord(c, ph, pw) ] );
	}
      }
    }
    if( previous_layer != nullptr )
      previous_layer->back_propagate();
  }

  void gradient_descent(F learning_rate, F momentum){
    if( next_layer != nullptr )
      next_layer->gradient_descent(learning_rate, momentum);
  }

  int stride;
  int pooling_size;
private:
  const F inf = 1e9;
  std::vector< std::pair<int,int> > unit_max_coord;
};

#endif
