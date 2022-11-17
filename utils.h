//
// Created by ljx on 2022/11/10.
//
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

#ifndef LAC_GWCNET_UTILS_H
#define LAC_GWCNET_UTILS_H


void ReadImages(const std::string imagesPath, std::vector<std::string> &lImg, std::vector<std::string> &rImg);

bool isContain(std::string str1, std::string str2);

void mat_to_vector(cv::Mat in,std::vector<float> &out);

void vector_to_mat(std::vector<float> in, cv::Mat out,int cols , int rows);

void ReadImages(const std::string imagesPath, std::vector<std::string> &lImg, std::vector<std::string> &rImg);
#endif //LAC_GWCNET_UTILS_H
