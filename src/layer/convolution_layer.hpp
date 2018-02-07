#ifndef CONVLUTIONLAYER
#define CONVLUTIONLAYER
#include "layer_base.hpp"
#include "layer_2d.hpp"

class ConvolutionLayer : public Layer2D {
  // stride = 1
public:
  ConvolutionLayer(){ }
  ConvolutionLayer(int ch, int fs, Layer * prev, int pch, int ph, int pw, ActivationFunction * af, std::string ln) {
    filter_size = fs;
    prev_channel = pch;
    prev_h = ph;
    prev_w = pw;
    if( prev->units != prev_channel * prev_h * prev_w ){
      throw "not compatible layer size";
    }
    unit_h = prev_h - 2 * ( filter_size / 2 );
    unit_w = prev_w - 2 * ( filter_size / 2 );
    init( channel * unit_h * unit_w, prev, af, "convolution : " + ln );
    init_conv();
  }
  ConvolutionLayer(int ch, int fs, Layer2D * prev, ActivationFunction * af, std::string ln) {
    channel = ch;
    filter_size = fs;
    unit_h = prev->unit_h - 2 * ( filter_size / 2 );
    unit_w = prev->unit_h - 2 * ( filter_size / 2 );
    prev_channel = prev->channel;
    prev_h = prev->unit_h;
    prev_w = prev->unit_w;
    init( channel * unit_h * unit_w, prev, af, "[convolution]" + ln );
    init_conv();
  }
  virtual void propagate(){
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
          activated_output[ idx ] = activation_func->f( unit_output[ idx ] );
        }
      }
    }
    if( next_layer != nullptr )
      next_layer->propagate();
  }
  virtual void back_propagate(){
    // compute previous layer's delta
    vec & prev_delta = previous_layer->delta;
    std::fill( prev_delta.begin(), prev_delta.end(), 0 );
    for(int ch = 0; ch < channel; ch++){
      for(int pch = 0; pch < prev_channel; pch++){
        for(int h = 0; h < unit_h; h++){
          for(int w = 0; w < unit_w; w++){
            for(int p = 0; p < filter_size; p++){
              for(int q = 0; q < filter_size; q++){
                prev_delta[ prev_coord(pch, h + p, w + q) ]
                += delta[ unit_coord(ch, h, w) ]
                * filter[ filter_coord(ch, pch, p, q) ]
                * previous_layer->activation_func->df( previous_layer->unit_output[ prev_coord(pch, h + p, w + q) ] );
              }
            }
          }
        }
      }
    }
    if( previous_layer != nullptr )
      previous_layer->back_propagate();
  }
  virtual void gradient_descent(F learning_rate, F momentum){
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
    update_bias( learning_rate, momentum );
    if( next_layer != nullptr )
      next_layer->gradient_descent(learning_rate, momentum);
  }
  
  int filter_size;

protected:
  vec bias;
  vec dbias;
  vec sum_square_grad_bias;
  vec filter;
  vec dfilter;
  vec sum_square_grad_filter;

  void update_bias(F learning_rate, F momentum){
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
  }
  int filter_coord( int tc, int pc, int s, int t ){
    return tc * prev_channel * filter_size * filter_size + pc * filter_size * filter_size + s * filter_size + t;
  }
  bool is_in_filter(int tc, int pc, int s, int t ){
    return (0 <= tc && tc < channel
	    && 0 <= pc && pc < prev_channel
	    && 0 <= s && s < filter_size
	    && 0 <= t && t < filter_size );
  }
  void init_conv(){
    int filter_total = channel * prev_channel * filter_size * filter_size;
    filter.resize( filter_total );
    dfilter.resize( filter_total, 0 );
    sum_square_grad_filter.resize( filter_total, 0 );

    bias.resize(channel, 0);
    dbias.resize(channel, 0);
    sum_square_grad_bias.resize(channel, 0 );

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
