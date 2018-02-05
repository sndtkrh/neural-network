#ifndef CONVLUTIONZEROPADDINGLAYER
#define CONVLUTIONZEROPADDINGLAYER
#include "convolution_layer.hpp"

class ConvolutionZeroPaddingLayer : public ConvolutionLayer {
  // stride = 1
public:
  ConvolutionZeroPaddingLayer(int ch, int fs, Layer * prev, int pch, int ph, int pw, ActivationFunction af, std::string ln) {
    channel = ch;
    filter_size = fs;
    prev_channel = pch;
    prev_h = ph;
    prev_w = pw;
    if( prev->units != prev_channel * prev_h * prev_w ){
      throw "not compatible layer size";
    }
    unit_h = prev_h;
    unit_w = prev_w;
    init( channel * unit_h * unit_w, prev, af, "[convolution zero padding]" + ln );
    init_conv();
  }
  ConvolutionZeroPaddingLayer(int ch, int fs, Layer2D * prev, ActivationFunction af, std::string ln) {
    channel = ch;
    filter_size = fs;
    prev_channel = prev->channel;
    prev_h = prev->unit_h;
    prev_w = prev->unit_w;
    unit_h = prev_h;
    unit_w = prev_w;
    init( channel * unit_h * unit_w, prev, af, "[convolution zero padding]" + ln );
    init_conv();
  }
  void propagate(){
    vec & z = previous_layer->activated_output;
    for(int ch = 0; ch < channel; ch++){
      for(int h = 0; h < unit_h; h++){
        for(int w = 0; w < unit_w; w++){
          int idx = unit_coord(ch, h, w);
          unit_output[ idx ] = bias[ ch ];
          for(int pch = 0; pch < prev_channel; pch++){
            for(int s = 0; s < filter_size; s++){
              for(int t = 0; t < filter_size; t++){
		int p = s - filter_size / 2;
		int q = t - filter_size / 2;
		if( is_in_prev( pch, h + p, w + q ) ){
		  unit_output[ idx ] += z[ prev_coord( pch, h + p, w + q ) ] * filter[ filter_coord(ch, pch, s, t) ];
		}
              }
            }
          }
          activated_output[ idx ] = activation_func.f( unit_output[ idx ] );
        }
      }
    }
    if( next_layer != nullptr )
      next_layer->propagate();
  }
  void back_propagate(){
    // compute previous layer's delta
    vec & prev_delta = previous_layer->delta;
    fill( prev_delta.begin(), prev_delta.end(), 0 );
    for(int ch = 0; ch < channel; ch++){
      for(int pch = 0; pch < prev_channel; pch++){
        for(int h = 0; h < unit_h; h++){
          for(int w = 0; w < unit_w; w++){
            for(int s = 0; s < filter_size; s++){
              for(int t = 0; t < filter_size; t++){
		int p = s - filter_size / 2;
		int q = t - filter_size / 2;
		if( is_in_prev( pch, h + p, w + q ) ){
		  prev_delta[ prev_coord(pch, h + p, w + q) ]
		    += delta[ unit_coord(ch, h, w) ]
		    * filter[ filter_coord(ch, pch, s, t) ]
		    * previous_layer->activation_func.df( previous_layer->unit_output[ prev_coord(pch, h + p, w + q) ] );
		}
              }
            }
          }
        }
      }
    }
    if( previous_layer != nullptr )
      previous_layer->back_propagate();
  }
  void gradient_descent(F learning_rate, F momentum){
    // update filter weight
    for(int ch = 0; ch < channel; ch++){
      for(int pch = 0; pch < prev_channel; pch++){
        for(int s = 0; s < filter_size; s++){
          for(int t = 0; t < filter_size; t++){
            int p = s - filter_size / 2;
            int q = t - filter_size / 2;
            // update filter[ch][pch][s][t]
            int filter_idx = filter_coord(ch, pch, s, t);
            F grad = 0;
            for(int h = 0; h < unit_h; h++){
              for(int w = 0; w < unit_w; w++){
                if( is_in_prev( pch, h + p, w + q ) ){
                  grad += delta[ unit_coord(ch, h, w) ] * previous_layer->activated_output[ prev_coord(pch, h + p, w + q) ];
                }
              }
            }
            // AdaGrad
            sum_square_grad_filter[ filter_idx ] += grad * grad;
            dfilter[ filter_idx ] = - learning_rate * grad / std::sqrt( std::max(sum_square_grad_filter[ filter_idx ], (F)1.0) ) + momentum * dfilter[ filter_idx ];
            filter[ filter_idx ] += dfilter[ filter_idx ];
          }
        }
      }
    }
    // update bias
    update_bias( learning_rate, momentum );
    if( next_layer != nullptr )
      next_layer->gradient_descent(learning_rate, momentum);
  }
};

#endif
