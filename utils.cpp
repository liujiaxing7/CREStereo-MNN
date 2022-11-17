//
// Created by ljx on 2022/11/10.
//

#include "utils.h"
#include "opencv2/opencv.hpp"
bool isContain(std::string str1, std::string str2)
{
    if (str1.find(str2)!= std::string::npos)
    {
        return true;
    } else
    {
        return false;
    }

}

void ReadImages(const std::string imagesPath, std::vector<std::string> &lImg, std::vector<std::string> &rImg)
{
    std::vector<cv::String> allFiles;
    cv::glob(imagesPath, allFiles, true);

    if (allFiles.size() == 0)
    {
        std::cout<< "there in no file;"<<std::endl;
    }
    for (int i =0 ; i < allFiles.size(); ++i)
    {
        std::string filename = allFiles[i];
        if (isContain(allFiles[i],"L.png"))
        {
            lImg.push_back(filename);
        }
        else if (isContain(allFiles[i], "R.png"))
        {
            rImg.push_back(filename);
        }
    }

    std::cout<<"load images"<<std::endl;
}

void mat_to_vector(cv::Mat in,std::vector<float> &out){

    for (int i=0; i < in.rows; i++) {
        for (int j =0; j < in.cols; j++){
            //unsigned char temp;

            //file << Dst.at<float>(i,j)  << endl;
            out.push_back(in.at<float>(i,j));
        }
    }

}

void vector_to_mat(std::vector<float> in, cv::Mat out,int cols , int rows){
    for (int i=rows-1; i >=0; i--) {
        for (int j =cols -1; j >=0; j--){

            out.at<float>(i,j) = in.back();
            in.pop_back();
            //file << Dst.at<float>(i,j)  << endl;
            // dst_temp.push_back(Dst.at<float>(i,j));
        }
    }
}