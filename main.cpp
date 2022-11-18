//
// Created by ljx on 2022/11/10.
//
#include <iostream>
#include "utils.h"
#include "string.h"
#include "Interpreter.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Expr.hpp>


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

//    const std::vector<std::string> input_names{"input.1", "input.401"};
//    const std::vector<std::string> output_names{"16382"};

    const std::vector<std::string> input_names{"left", "right"};
    const std::vector<std::string> output_names{"output"};

//    auto type = MNN_FORWARD_OPENCL;
//    MNN::ScheduleConfig Sconfig;
//    Sconfig.type      = type;
//    Sconfig.numThread = 4;
//    Sconfig.backupType = type;
//    MNN::BackendConfig backendConfig;
//    int precision = MNN::BackendConfig::Precision_Normal;
//    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
//    Sconfig.backendConfig     = &backendConfig;

    MNN::Express::Module::Config mConfig;
//    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr(MNN::Express::Executor::RuntimeManager::createRuntimeManager(Sconfig));
//    std::shared_ptr<MNN::Express::Module> net(
//            MNN::Express::Module::load(input_names, output_names, mnn_path.c_str(),rtmgr,  &mConfig));
    std::unique_ptr<MNN::Express::Module> module;
    module.reset(MNN::Express::Module::load(input_names, output_names, mnn_path.c_str(), &mConfig));

    auto info = module->getInfo();


    std::string imagespath = argv[2];

    std::vector<std::string> limg;
    std::vector<std::string> rimg;


    ReadImages(imagespath, limg, rimg);


    for (size_t i = 0; i < limg.size(); ++i) {

        int w = 320;
        int h = 240;
        int c = 3;

        auto inputLeft = MNN::Express::_Input({1, 3, h, w}, MNN::Express::NC4HW4, halide_type_of<float>());
        auto inputRight = MNN::Express::_Input({1, 3, h, w}, MNN::Express::NC4HW4, halide_type_of<float>());

        auto imgL = limg.at(i);
        auto imgR = rimg.at(i);


        int width, height, channel;
        auto imageL = stbi_load(imgL.c_str(), &width, &height, &channel, 4);
        auto imageR = stbi_load(imgR.c_str(), &width, &height, &channel, 4);


//        cv::Mat gray1_mat(375, 450, CV_8UC3, imageR);
//        imshow("去雾图像显示", gray1_mat);
//        cv::waitKey();
//        std::cout << "load images success" << std::endl;


        MNN::CV::Matrix trans;
        trans.setScale((float)(width-1) / (w-1), (float)(height-1) / (h-1));

        MNN::CV::ImageProcess::Config config;
        config.filterType = MNN::CV::BILINEAR;
        config.sourceFormat = MNN::CV::RGBA;
        config.destFormat = MNN::CV::BGR;

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config));
        pretreat->setMatrix(trans);

        pretreat->convert((uint8_t *) imageL, width, height, 0, inputLeft->writeMap<float>() + 0 * 4 * w * h , w, h,
                          4, 0, halide_type_of<float>());

        std::shared_ptr<MNN::CV::ImageProcess> pretreat1(MNN::CV::ImageProcess::create(config));
        pretreat1->setMatrix(trans);


        pretreat1->convert((uint8_t *) imageR, width, height, 0, inputRight->writeMap<float>() + 0 * 4 * w * h , w, h,
                          4, 0, halide_type_of<float>());





        std::cout << "forward" << std::endl;
//        inputLeft->resize({1, 3, 240,320});
//        inputRight->resize({1, 3, 240,320});


        MNN::Express::Executor::getGlobalExecutor()->resetProfile();
        auto outputs = module->onForward({inputLeft, inputRight});
        auto output = MNN::Express::_Convert(outputs[0], MNN::Express::NHWC);
        output = MNN::Express::_Reshape(output, {2, 240, 320});
        auto value = output->readMap<float>();


        //output赋值给mat
        int outSize_w = 320;
        int outSize_h = 240;
        cv::Mat outimg;
        outimg.create(cv::Size(outSize_w, outSize_h), CV_32FC1);

        cv::Mat showImg;

        for (int i=0; i<outSize_h; ++i) {
            {
                for (int j=0; j<outSize_w; ++j)
                {
                    outimg.at<float>(i,j) = value[i*outSize_w+j];
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
        cv::waitKey(0);

//        cv::Mat saveImg;
//        showImg.convertTo(saveImg,CV_8UC4);
//        cv::imwrite("a.png", saveImg);

        std::cout << "success" << std::endl;

    }
    return 0;
}