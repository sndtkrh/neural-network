#ifndef CONVLUTIONLAYER
#define CONVLUTIONLAYER
#include "layer_base.hpp"

class ConvolutionLayer : public Layer {
  // stride = 1
public:
  ConvolutionLayer(int channel, int filter_size, Layer * prev, int prev_channel, int prev_h, int prev_w, ActivationFunction af, std::string ln)
  : channel(channel), filter_size(filter_size), prev_channel(prev_channel), prev_h(prev_h), prev_w(prev_w) {
    if( prev->units != prev_channel * prev_h * prev_w ){
      throw "not compatible layer size";
    }
    unit_h = prev_h - 2 * ( filter_size / 2 );
    unit_w = prev_w - 2 * ( filter_size / 2 );
    init( channel * unit_h * unit_w, prev, af, "convolution : " + ln );
    init_conv();
  }
  ConvolutionLayer(int channel, int filter_size, ConvolutionLayer * prev, ActivationFunction af, std::string ln)
  : channel(channel), filter_size(filter_size) {
    unit_h = prev->unit_h - 2 * ( filter_size / 2 );
    unit_w = prev->unit_h - 2 * ( filter_size / 2 );
    prev_channel = prev->channel;
    prev_h = prev->unit_h;
    prev_w = prev->unit_w;
    init( channel * unit_h * unit_w, prev, af, "convolution : " + ln );
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
            for(int p = 0; p < filter_size; p++){
              for(int q = 0; q < filter_size; q++){
                unit_output[ idx ] += z[ prev_coord( pch, h + p, w + q ) ] * filter[ filter_coord(ch, pch, p, q) ];
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
            for(int p = 0; p < filter_size; p++){
              for(int q = 0; q < filter_size; q++){
                prev_delta[ prev_coord(pch, h + p, w + q) ]
                += delta[ unit_coord(ch, h, w) ]
                * filter[ filter_coord(ch, pch, p, q) ]
                * previous_layer->activation_func.df( previous_layer->unit_output[ prev_coord(pch, h + p, w + q) ] );
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
        for(int p = 0; p < filter_size; p++){
          for(int q = 0; q < filter_size; q++){
            // update filter[ch][pch][p][q] here
            int filter_idx = filter_coord(ch, pch, p, q);
            F grad = 0;
            for(int h = 0; h < unit_h; h++){
              for(int w = 0; w < unit_w; w++){
                grad += delta[ unit_coord(ch, h, w) ] * previous_layer->activated_output[ prev_coord(pch, h + p, w + q) ];
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
    for(int ch = 0; ch < channel; ch++){
      F grad = 0;
      for(int h = 0; h < unit_h; h++){
        for(int w = 0; w < unit_w; w++){
          grad += delta[ unit_coord( ch, h, w ) ];
        }
      }
      // AdaGrad
      sum_square_grad_bias[ ch ] += grad * grad;
      dbias[ ch ] = - learning_rate * grad / std::sqrt( std::max(sum_square_grad_bias[ ch ], (F)1.0 ) ) + momentum * dbias[ ch ];
      bias[ ch ] += dbias[ ch ];
    }
    if( next_layer != nullptr )
      next_layer->gradient_descent(learning_rate, momentum);
  }
  
  int filter_size;
  int channel, unit_h, unit_w;
  int prev_channel, prev_h, prev_w;
  
private:
  vec bias;
  vec dbias;
  vec sum_square_grad_bias;
  vec filter;
  vec dfilter;
  vec sum_square_grad_filter;
  vec delta;
  
  int unit_coord( int c, int y, int x ){
    return c * unit_h * unit_w + y * unit_w + x;
  }
  int prev_coord( int c, int y, int x ){
    return c * prev_h * prev_w + y * prev_w + x;
  }
  int filter_coord( int tc, int pc, int y, int x ){
    return tc * prev_channel * filter_size * filter_size + pc * filter_size * filter_size + y * filter_size + x;
  }
  void init_conv(){
    filter.resize(channel * prev_channel * filter_size * filter_size);
    dfilter.resize(channel * prev_channel * filter_size * filter_size, 0);
    sum_square_grad_filter.resize(prev_channel * filter_size * filter_size, 0);
    
    bias.resize(channel, 0);
    dbias.resize(channel, 0);
    sum_square_grad_bias.resize(channel, 0);
    
    delta.resize( channel * unit_h * unit_w );
    
    // random initialization
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::normal_distribution<> dist(0.0, 0.1);
    for(int i = 0; i < filter.size(); i++){
      filter[i] = dist( engine );
    }
  }
};

#endif
