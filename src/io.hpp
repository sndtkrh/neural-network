#include <iostream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "common.hpp"
#include "matrix.hpp"

std::vector<std::string> enum_filenames(const std::string path);
vec mat_to_vec( cv::Mat m );

void load_dataset(std::string dataset_dir, std::vector<std::vector<vec> > & dataset, int size){
  dataset.resize(10);
  for(int i = 0; i < 10; i++){
    std::vector<std::string> mnist_dataset_filenames;
    mnist_dataset_filenames = enum_filenames( dataset_dir + "/" + std::to_string(i) + "/");
    for( std::string f : mnist_dataset_filenames ){
      if( size != -1 && dataset[i].size() >= size ){
	break;
      }
      dataset[i].push_back( mat_to_vec( cv::imread( f, 0 ) ) );
    }
  }
}
void load_dataset(std::string dataset_dir, std::vector<std::vector<vec> > & dataset){
  load_dataset(dataset_dir, dataset, -1);
}

void save_image( std::string filename, std::vector<F> v, int h, int w ){
  std::cout << "saving image..." << std::endl;
  cv::Mat image = cv::Mat::zeros( h, w, CV_8UC1);
  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      image.at<uchar>(i, j) = (uchar)( std::min(v[ i * w + j ] * 255.0, 254.9) );
    }
  }
  cv::imwrite(filename, image);
}

vec mat_to_vec( cv::Mat m ){
  vec v( m.rows * m.cols, 0 );
  std::cout << "mat_to_vec " << m.rows << "x" << m.cols << std::endl;
  for(int i = 0; i < m.cols; i++){
    for(int j = 0; j < m.rows; j++){
      v[ i * m.cols + j ] = (F)m.at<uchar>( i, j ) / 255.0;
    }
  }
  return v;
}

std::vector<std::string> enum_filenames(const std::string path){
  std::vector<std::string> filenames;
  DIR *dp;
  dirent* entry;
    
  dp = opendir(path.c_str());
  if( dp == NULL ) return filenames;
  do {
    entry = readdir(dp);
    if (entry != NULL && entry->d_name[0] != '.' )
      filenames.push_back( path + "/" + entry->d_name );
  } while (entry != NULL);
  return filenames;
}
