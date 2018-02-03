#include <iostream>
#include <string>
#include <random>
#include "matrix.hpp"
#include "io.hpp"
#include "layers.hpp"

const std::string TRAINING_DATASET_DIR = "../MNIST_dataset/mnist_png/training";
const std::string TESTING_DATASET_DIR = "../MNIST_dataset/mnist_png/testing";

const int IMAGE_H = 28;
const int IMAGE_W = 28;
std::vector<std::vector<vec> > mnist_dataset[2]; // 0:training, 1:testing

void one_step( InputLayer & input, Layer & output, vec data, vec target );
void test( InputLayer & input, SoftmaxLayer & output );

int main(){
  std::random_device rnd;
  std::mt19937 mt(rnd());

  load_dataset_max100(TRAINING_DATASET_DIR, mnist_dataset[0]);
  load_dataset_max100(TESTING_DATASET_DIR, mnist_dataset[1]);
  std::cout << "loaded" << std::endl;

  // construct neural network
  ActivationFunction rel = ReL();
  InputLayer input( IMAGE_H * IMAGE_W );
  ConvolutionLayer conv1( 10, 3, &input, 1, IMAGE_H, IMAGE_W, rel, "1" );
  ConvolutionLayer conv2( 5, 3, &conv1, rel, "2" );
  FullyConnectedLayer full1( 30, &conv2, rel, "1" );
  SoftmaxLayer softmax( 10, &full1 );

  input.print_info();
  conv1.print_info();
  conv2.print_info();
  full1.print_info();
  softmax.print_info();
  std::cout << "constructed" << std::endl;
  
  vec image; // = mnist_dataset[0][0][0];
  vec target(10,0);// target[0] = 0;

  // learning
  for(int i = 0; i < 10000; i++){
    for(int j = 0; j < 10; j++){
      std::uniform_int_distribution<> rand(0, mnist_dataset[0][j].size()-1 );
      image = mnist_dataset[0][j][ rand(mt) ];
      target[j] = 1.0;
      one_step( input, softmax, image, target );
      target[j] = 0;
    }
    if( i % 100 == 0 ){
      std::cout << "i=" << i << std::endl;
      test( input, softmax );
    }
  }
  std::cout << "[[[[ learned ]]]]" << std::endl;

  // testing
  test( input, softmax );
}

void one_step( InputLayer & input, Layer & output, vec data, vec target ){
  //std::cout << "propagate" << std::endl;
  input.propagate( data );

  //std::cout << "back propagate" << std::endl;
  output.set_target( target );
  output.back_propagate( );
  
  //std::cout << "grad descent" << std::endl;
  input.gradient_descent( 0.01, 0.5 );
  //std::cout << "end" << std::endl;
  //print_vec( output.activated_output );
}

void test( InputLayer & input, SoftmaxLayer & output ){
  int n = 0;
  int correct = 0;
  vec image;

  for(int i = 0; i < 10; i++){
    for(int j = 0; j < mnist_dataset[1][i].size(); j++){
      image = mnist_dataset[1][i][j];
      input.propagate( image );
      if( i == output.get_class() ){
	correct++;
      }
      n++;
    }
  }
  std::cout << "total test data size = " << n << std::endl;
  std::cout << "correct answer = " << correct << std::endl;
  std::cout << "rate = " << 1.0 * correct / n << std::endl;
}
