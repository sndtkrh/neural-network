#include <iostream>
#include "src/neuralnetwork.hpp"

const std::string TRAINING_DATASET_DIR = "../MNIST_dataset/mnist_png/training";
const std::string TESTING_DATASET_DIR = "../MNIST_dataset/mnist_png/testing";
std::vector<std::vector<vec> > train_data;
std::vector<std::vector<vec> > test_data;

const int IMAGE_H = 28;
const int IMAGE_W = 28;

void one_step( InputLayer2D & input, Layer & output, vec image );
void test( InputLayer2D & input, Layer & output, int i );

int main(){
  std::random_device rnd;
  std::mt19937 mt(rnd());
  
  load_dataset(TRAINING_DATASET_DIR, train_data, 1000);
  load_dataset(TESTING_DATASET_DIR, test_data, 1);
  std::cout << "[[[ loaded ]]]" << std::endl;
  std::cout << std::endl;
  for(int i = 0; i < 10; i++){
    save_image( "output/train" + std::to_string(i) + ".png", test_data[i][0], IMAGE_H, IMAGE_W );
  }

  // construct autoencoder
  std::cout << relu.f( -1.0 ) << std::endl;

  InputLayer2D input( 1, IMAGE_H, IMAGE_W );
  FullyConnectedLayer med( 50, &input, &relu, "med" );
  FullyConnectedLayer output( IMAGE_H * IMAGE_W, &med, &sigmoid, "output" );
  input.print_network_info();
  std::cout << "[[[ constructed ]]]" << std::endl;
  std::cout << std::endl;

  // learning
  vec image;
  for(int i = 0; i < 10000; i++){
    for(int j = 0; j < 10; j++){
      std::uniform_int_distribution<> rand(0, train_data[j].size()-1 );
      image = train_data[j][ rand(mt) ];
      one_step( input, output, image );
    }
    if( i % 1000 == 0 ){
      std::cout << "i=" << i << std::endl;
      test( input, output, i );
    }
  }
}

void one_step( InputLayer2D & input, Layer & output, vec image ){
  input.propagate( image );
  output.set_target( image );
  output.back_propagate();
  input.gradient_descent(0.01, 0.5);
}

void test( InputLayer2D & input, Layer & output, int i ){
  for(int j = 0; j < 10; j++){
    input.propagate( test_data[j][0] );
    print_vec( input.next_layer->unit_output );
    print_vec( input.next_layer->activated_output );
    save_image( "output/test" + std::to_string(i) + "_" + std::to_string(j) + ".png", output.activated_output, IMAGE_H, IMAGE_W );
    F e = 0;
    for(int u = 0; u < IMAGE_H * IMAGE_W; u++){
      F d = output.activated_output[u] - test_data[j][0][u];
      e += 0.5 * d * d;
    }
    std::cout << "E=" << e << std::endl;
  }
}
