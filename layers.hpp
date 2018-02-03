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
  ActivationFunction func(){
    return activation_func;
  }
  std::string name(){
    return layer_name;
  }

  void set_target( vec & t ){
    target = t;
  }

  void print_info( ){
    std::cout << "layer name = " << layer_name << std::endl;
    std::cout << "inputs=" << inputs << std::endl;
    std::cout << "units=" << units << std::endl;
    if( previous_layer != nullptr )
      std::cout << "previous_layer=" << previous_layer->layer_name << std::endl;
    if( next_layer != nullptr )
      std::cout << "next_layer=" << next_layer->layer_name << std::endl;
    std::cout << std::endl;
  }
protected:
  vec target;
  void init( int u, Layer * prev, ActivationFunction af, std::string ln) {
    next_layer = nullptr;
    previous_layer = prev;
    previous_layer->next_layer = this;
    units = u;
    inputs = prev->units;
    activation_func = af;
    layer_name = ln;
    unit_output.resize( units );
    activated_output.resize( units );
    delta.resize( units, 0 );
  }
};


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

class FullyConnectedLayer : public Layer {
public:
  FullyConnectedLayer(int u, Layer * prev, ActivationFunction af, std::string ln){
    init( u, prev, af, "fully connected : " + ln );
    weight.resize( units );
    dweight.resize( units );
    sum_square_grad.resize( units );
    for(int i = 0; i < units; i++){
      weight[i].resize( inputs );
      dweight[i].resize( inputs );
      sum_square_grad[i].resize( inputs, 0 );
    }
    bias.resize( units, 0 );
    dbias.resize( units, 0 );

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
    compute_previous_layer_delta();   
    if( previous_layer != nullptr )
      previous_layer->back_propagate();
  }
  void compute_previous_layer_delta(){
    vec & prev_delta = previous_layer->delta;
    std::fill( prev_delta.begin(), prev_delta.end(), 0 );
    for(int pu = 0; pu < inputs; pu++){
      for(int u = 0; u < units; u++){
	prev_delta[pu] += delta[u] * weight[u][pu] * previous_layer->activation_func.df( previous_layer->unit_output[pu] );
      }
    }
  }
  virtual void gradient_descent( F learning_rate, F momentum ){
    vec & z = previous_layer->activated_output;
    for(int i = 0; i < units; i++){
      for(int j = 0; j < inputs; j++){
	F grad = delta[i] * z[j];
	// AdaGrad
	sum_square_grad[i][j] += grad * grad;
	dweight[i][j] = - learning_rate * grad / std::sqrt( std::max(sum_square_grad[i][j], (F)1.0) ) + momentum * dweight[i][j];
	weight[i][j] += dweight[i][j];
      }
    }
    for(int i = 0; i < units; i++){
      dbias[i] = - learning_rate * delta[i] + momentum * dbias[i];
      bias[i] += dbias[i];
    }
    if( next_layer != nullptr ){
      next_layer->gradient_descent( learning_rate, momentum );
    }
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

class InputLayer : public Layer {
public:
  InputLayer( int u ){
    units = u;
    unit_output.resize( units );
    delta.resize( units );
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

private:
  vec input_vec;
};

#endif
