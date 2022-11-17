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
//        std::vector<MNN::Express::VARP> inputs(2);
//        inputs[0] =  MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NC4HW4);
//        inputs[1] =  MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NC4HW4);
        auto inputLeft = MNN::Express::_Input({1, 3, h, w}, MNN::Express::NC4HW4, halide_type_of<float>());

        auto inputRight = MNN::Express::_Input({1, 3, h, w}, MNN::Express::NC4HW4, halide_type_of<float>());
        //auto other = MNN::Express::_Input({1, 3, 400, 640}, MNN::Express::NC4HW4);

        auto imgL = limg.at(i);
        auto imgR = rimg.at(i);


        int width, height, channel;
        auto imageL = stbi_load(imgL.c_str(), &width, &height, &channel, 4);
        auto imageR = stbi_load(imgR.c_str(), &width, &height, &channel, 4);

        //std::cout<<imageR<<std::endl;
//        cv::Mat gray1_mat(400, 640, CV_8UC3, imageR);
//        imshow("去雾图像显示", gray1_mat);
//        cv::waitKey();

        std::cout << "read images" << std::endl;

//

        MNN::CV::Matrix trans;
        trans.setScale((float)(width-1) / (w-1), (float)(height-1) / (h-1));

        MNN::CV::ImageProcess::Config config;
        config.filterType = MNN::CV::BILINEAR;

//        float mean[3] = {103.94f, 116.78f, 123.68f};
//        float normals[3] = {0.017f, 0.017f, 0.017f};

//        ::memcpy(config.mean, mean, sizeof(mean));
//        ::memcpy(config.normal, normals, sizeof(mean));

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config));
        pretreat->setMatrix(trans);

        pretreat->convert((uint8_t *) imageL, width, height, 0, inputLeft->writeMap<float>() + 0 * 4 * w * h , w, h,
                          4, 0, halide_type_of<float>());

        std::shared_ptr<MNN::CV::ImageProcess> pretreat1(MNN::CV::ImageProcess::create(config));
        pretreat1->setMatrix(trans);


        pretreat1->convert((uint8_t *) imageR, width, height, 0, inputRight->writeMap<float>() + 0 * 4 * w * h , w, h,
                          4, 0, halide_type_of<float>());

        std::cout << "forward" << std::endl;
//        inputLeft->resize({1, 3, 400,640});

//
        MNN::Express::Executor::getGlobalExecutor()->resetProfile();

        auto outputs = module->onForward({inputLeft, inputRight});
        auto output = MNN::Express::_Convert(outputs[0], MNN::Express::NHWC);
        output = MNN::Express::_Reshape(output, {2, 240, 320});
        int topK = 2* w*h;
//        auto topKV = MNN::Express::_TopKV2(output, MNN::Express::_Scalar<float>(topK));
        auto value = output->readMap<float>();
//        auto indice = topKV[1]->readMap<int>();

//        std::vector<float> outimg;
        cv::Mat outimg;
        outimg.create(cv::Size(w, h), CV_32FC1);

        cv::Mat showImg;


        for (int i=0; i<h; ++i) {
            {
                for (int j=0; j<w; ++j)
                {
                    std::cout<<value[i*h+j]<<std::endl;
                    outimg.at<float>(i,j) = value[i*w+j];
                }
            }
        }

//        outimg.convertTo(showImg,CV_8U);
//        cv::namedWindow(
//                "MyWindow"
//                , CV_WINDOW_AUTOSIZE);

        double minv = 0.0, maxv = 0.0;
        double* minp = &minv;
        double* maxp = &maxv;
        minMaxIdx(outimg,minp,maxp);
        float minvalue = (float)minv;
        float maxvalue = (float)maxv;

        for (int i=0; i<h; ++i) {
            {
                for (int j=0; j<w; ++j)
                {

                    outimg.at<float>(i,j) = 255* (outimg.at<float>(i,j) - minvalue)/(maxvalue-minvalue);
                }
            }
        }

        namedWindow("image", cv::WINDOW_AUTOSIZE);
        imshow("image", outimg);
        cv::waitKey(0);
//        cv::imwrite("./a.png", outimg);
//        imshow(
//                "MyWindow"
//                , showImg);
//        cv::waitKey(0);
//
//        cv::destroyWindow(
//                "MyWindow"
//        );

//
//        std::cout<<outimg<<std::endl;
//        cv::Mat depthimg;
//
//        vector_to_mat(outimg,depthimg,320,240);


//        std::vector<float> outimg;
////        for (int batch = 0; batch < 1; ++batch) {
////            MNN_PRINT("For Input: %s \n", argv[batch+2]);
//        for (int i=0; i<topK; ++i) {
//            MNN_PRINT(" %f\n", value[0,0,topK + i]);
//            outimg.push_back(value[topK + i]);
//        }
////        }

        std::cout<<"final"<<std::endl;
//        MNN::Express::Executor::getGlobalExecutor()->dumpProfile();
        std::cout << "success" << std::endl;

    }
    return 0;
}