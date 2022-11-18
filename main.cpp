//
// Created by ljx on 2022/11/10.
//
#include <iostream>
#include "utils.h"
#include "string.h"
#include <opencv2/highgui.hpp>
#include <ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Expr.hpp>
#include<sys/time.h>
#include <MNN/expr/ExecutorScope.hpp>

#define STB_IMAGE_IMPLEMENTATION
//#define STBI_NO_STDIO
#include "stb_image.h"

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << "output_path:: output_dir" << std::endl;
        return -1;
    }

    const std::string mnn_path = argv[1];

    //计算加载时间
    float load_time_use = 0;
    struct timeval load_start;
    struct timeval load_end;
    gettimeofday(&load_start,NULL);

    const std::vector<std::string> input_names{"left", "right"};
    const std::vector<std::string> output_names{"output"};

    MNNForwardType type = MNN_FORWARD_CUDA;
    MNN::BackendConfig backend_config;    // default backend config
    int precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;
    backend_config.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
    std::shared_ptr<MNN::Express::Executor> executor(
            MNN::Express::Executor::newExecutor(type, backend_config, 4));
    MNN::Express::ExecutorScope scope(executor);

    MNN::Express::Module::Config mConfig;
    std::shared_ptr<MNN::Express::Module> module(
            MNN::Express::Module::load(input_names, output_names, mnn_path.c_str(), &mConfig));


    gettimeofday(&load_end,NULL);
    load_time_use=(load_end.tv_sec-load_start.tv_sec)*1000000+(load_end.tv_usec-load_start.tv_usec);
    std::cout<<"load model time : "<<load_time_use<<std::endl;

    std::string imagespath = argv[2];
    std::vector<std::string> limg;
    std::vector<std::string> rimg;
    ReadImages(imagespath, limg, rimg);

    for (size_t i = 0; i < limg.size(); ++i) {

        int w = 320;
        int h = 240;
        int c = 3;
        int outbpp = 4;

        auto inputLeft = MNN::Express::_Input({1, 3, h, w}, MNN::Express::NC4HW4, halide_type_of<float>());
        auto inputRight = MNN::Express::_Input({1, 3, h, w}, MNN::Express::NC4HW4, halide_type_of<float>());

        auto imgL = limg.at(i);
        auto imgR = rimg.at(i);

        int width, height, channel;
        auto imageL = stbi_load(imgL.c_str(), &width, &height, &channel, outbpp);
        auto imageR = stbi_load(imgR.c_str(), &width, &height, &channel, outbpp);

//        cv::Mat gray1_mat(375, 450, CV_8UC3, imageR);
//        imshow("去雾图像显示", gray1_mat);
//        cv::waitKey();
//        std::cout << "load images success" << std::endl;

        //data to gpu time
        float data_to_gpu = 0;
        struct timeval data_gpu_start;
        struct timeval data_gpu_end;
        gettimeofday(&data_gpu_start,NULL);

        MNN::CV::Matrix trans;
        trans.setScale((float)(width-1) / (w-1), (float)(height-1) / (h-1));

        MNN::CV::ImageProcess::Config config;
        config.filterType = MNN::CV::BILINEAR;
        config.sourceFormat = MNN::CV::RGBA;
        config.destFormat = MNN::CV::BGR;

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t *) imageL, width, height, 0, inputLeft->writeMap<float>() , w, h,
                          outbpp, 0, halide_type_of<float>());
        stbi_image_free(imageL);

        std::shared_ptr<MNN::CV::ImageProcess> pretreat1(MNN::CV::ImageProcess::create(config));
        pretreat1->setMatrix(trans);
        pretreat1->convert((uint8_t *) imageR, width, height, 0, inputRight->writeMap<float>() , w, h,
                           outbpp, 0, halide_type_of<float>());
        stbi_image_free(imageR);

        gettimeofday(&data_gpu_end,NULL);
        data_to_gpu=(data_gpu_end.tv_sec-data_gpu_start.tv_sec)*1000000+(data_gpu_end.tv_usec-data_gpu_start.tv_usec);
        std::cout<<"data_to_gpu time : "<<data_to_gpu<<std::endl;

        //计算forward时间
        float forward_time_use = 0;
        struct timeval forward_start;
        struct timeval forward_end;
        gettimeofday(&forward_start,NULL);

        std::cout << "forward" << std::endl;
        auto outputs = module->onForward({inputLeft, inputRight});

        gettimeofday(&forward_end,NULL);
        forward_time_use=(forward_end.tv_sec-forward_start.tv_sec)*1000000+(forward_end.tv_usec-forward_start.tv_usec);
        std::cout<<"forward time : "<<forward_time_use<<std::endl;

        //data to cpu time
        float data_to_cpu = 0;
        struct timeval data_cpu_start;
        struct timeval data_cpu_end;
        gettimeofday(&data_cpu_start,NULL);

        auto output = MNN::Express::_Convert(outputs[0], MNN::Express::NHWC);
        output = MNN::Express::_Reshape(output, {2, 240, 320});
        auto value = output->readMap<float>();

        gettimeofday(&data_cpu_end,NULL);
        data_to_cpu=(data_cpu_end.tv_sec-data_cpu_start.tv_sec)*1000000+(data_cpu_end.tv_usec-data_cpu_start.tv_usec);
        std::cout<<"data_to_cpu time : "<<data_to_cpu<<std::endl;

        //output赋值给mat
        int outSize_w = w;
        int outSize_h = h;
        cv::Mat outimg;
        outimg.create(cv::Size(outSize_w, outSize_h), CV_32FC1);

        cv::Mat showImg;

        for (int i=0; i<outSize_h; ++i) {
            {
                for (int j=0; j<outSize_w; ++j)
                {
                    outimg.at<float>(i,j) = value[(i*outSize_w+j)*2];
                }
            }
        }

        //可视化
        double minv = 0.0, maxv = 0.0;
        double* minp = &minv;
        double* maxp = &maxv;
        minMaxIdx(outimg,minp,maxp);
        float minvalue = (float)minv;
        float maxvalue = (float)maxv;

        for (int i=0; i<outSize_h; ++i) {
            {
                for (int j=0; j<outSize_w; ++j)
                {

                    outimg.at<float>(i,j) = 255* (outimg.at<float>(i,j) - minvalue)/(maxvalue-minvalue);
                }
            }
        }

        outimg.convertTo(showImg,CV_8U);
        namedWindow("image", cv::WINDOW_AUTOSIZE);
        imshow("image", showImg);
//        cv::waitKey(0);

        std::cout << "success" << std::endl;
    }
    return 0;
}