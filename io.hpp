#include <iostream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "common.hpp"
#include "matrix.hpp"

std::vector<std::string> enum_filenames(const std::string path);
vec mat_to_vec( cv::Mat m );

void load_dataset(std::string dataset_dir, std::vector<std::vector<vec> > & dataset){
  dataset.resize(10);
  for(int i = 0; i < 10; i++){
    std::vector<std::string> mnist_dataset_filenames;
    char c = '0' + i;
    mnist_dataset_filenames = enum_filenames( dataset_dir + "/" + c + "/");
    for( std::string f : mnist_dataset_filenames ){
      dataset[i].push_back( mat_to_vec( cv::imread( f, 0 ) ) );
    }
  }
}

void save_image( std::string filename, std::vector<F> v, int h, int w ){
  std::cout << "saving image..." << std::endl;
  cv::Mat image = cv::Mat::zeros( h, w, CV_8UC1);
  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      image.at<uchar>(i, j) = (uchar)( std::max(v[ h * i + j ] * 255.0, 254.9) );
    }
  }
  /*
  cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
  cv::imshow("image", image);
  cv::waitKey(3000);
  */
  cv::imwrite(filename, image);
}

vec mat_to_vec( cv::Mat m ){
  vec v( m.rows * m.cols, 0 );
  for(int i = 0; i < m.rows; i++){
    for(int j = 0; j < m.cols; j++){
      v[ i * m.rows + j ] = (F)m.at<uchar>( i, j ) / 255.0;
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
