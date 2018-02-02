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

  load_dataset(TRAINING_DATASET_DIR, mnist_dataset[0]);
  load_dataset(TESTING_DATASET_DIR, mnist_dataset[1]);
  std::cout << "loaded" << std::endl;

  // construct neural network
  ActivationFunction rel = ReL();
  InputLayer input( IMAGE_H * IMAGE_W );
  FullyConnectedLayer full1( 100, &input, rel, "1" );
  FullyConnectedLayer full2( 50, &full1, rel, "2" );
  FullyConnectedLayer full3( 30, &full2, rel, "3" );
  SoftmaxLayer softmax( 10, &full3 );

  vec image;
  vec target(10,0);

  // learning
  for(int i = 0; i < 50000; i++){
    for(int j = 0; j < 10; j++){
      std::uniform_int_distribution<> rand(0, mnist_dataset[0][j].size()-1 );
      image = mnist_dataset[0][j][ rand(mt) ];
      target[j] = 1.0;
      one_step( input, softmax, image, target );
      target[j] = 0;
    }
    if( i % 1000 == 0 ){
      std::cout << "i=" << i << std::endl;
      test( input, softmax );
    }
  }
  std::cout << "[[[[ learned ]]]]" << std::endl;

  // testing
  test( input, softmax );
}

void one_step( InputLayer & input, Layer & output, vec data, vec target ){
  input.propagate( data );

  output.set_target( target );
  output.back_propagate( );
  
  input.gradient_descent( 0.01, 0.5 );
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
