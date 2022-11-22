#include <iostream>
#include "utils.h"
#include "string.h"
#include "Interpreter.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ImageProcess.hpp>
#include <sys/time.h>

int main(int argc, char **argv) {
//    std::cout << "Hello, World!" << std::endl;
    if (argc < 3) {
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images.txt\n"
                  << "output_path:: output_dir" << std::endl;
        return -1;
    }
    //计算加载时间
    float load_time_use = 0;
    struct timeval load_start;
    struct timeval load_end;
    gettimeofday(&load_start,NULL);

    const std::string mnn_path = argv[1];
    std::shared_ptr<MNN::Interpreter> my_interpreter = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromFile(mnn_path.c_str()));

    // config
    MNN::ScheduleConfig config;
    int num_thread = 4;
    config.numThread = num_thread;
//    config.mode = MNN_GPU_TUNING_NONE;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;
    int forward = MNN_FORWARD_OPENCL;
    config.type = static_cast<MNNForwardType>(forward);

    // create session
    MNN::Session *my_session = my_interpreter->createSession(config);
    my_interpreter->releaseModel();

    gettimeofday(&load_end,NULL);
    load_time_use=(load_end.tv_sec-load_start.tv_sec)*1000000+(load_end.tv_usec-load_start.tv_usec);
    std::cout<<"load model time : "<<load_time_use/1000.0<<"ms"<<std::endl;

    int w = 320;
    int h = 240;
    MNN::Tensor *input_tensorL = my_interpreter->getSessionInput(my_session, "left");
    my_interpreter->resizeTensor(input_tensorL, {1, 3, h, w});
    MNN::Tensor *input_tensorR = my_interpreter->getSessionInput(my_session, "right");
    my_interpreter->resizeTensor(input_tensorR, {1, 3, h, w});

//    my_interpreter->releaseSession(my_session);
//    my_interpreter->resizeTensor(input_tensor, {1, 3, 416, 416});

    std::string imagespath = argv[2];

    std::vector<std::string> limg;
    std::vector<std::string> rimg;

    ReadImages(imagespath, limg, rimg);

    for (size_t i =0 ; i < limg.size(); ++i)
    {
        auto imgL = limg.at(i);
        auto imgR = rimg.at(i);

        cv::Mat imginL = cv::imread(imgL);
        cv::Mat imginR = cv::imread(imgR);

        cv::Mat frameL = cv::Mat(imginL.rows, imginL.cols, CV_8UC3, imginL.data);
        cv::Mat frameR = cv::Mat(imginR.rows, imginR.cols, CV_8UC3, imginR.data);

        cv::Mat imageL, imageR;
        cv::resize(frameL, imageL, cv::Size(w, h), cv::INTER_LINEAR);
        cv::resize(frameR, imageR, cv::Size(w, h), cv::INTER_LINEAR);

        const float mean_vals[3] = {0.485, 0.456, 0.406};
        const float norm_vals[3] = {0.229, 0.224, 0.225};

        //data to gpu time
        float data_to_gpu = 0;
        struct timeval data_gpu_start;
        struct timeval data_gpu_end;
        gettimeofday(&data_gpu_start,NULL);
        std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::RGB));
        pretreat->convert(imageL.data, w, h, imageL.step[0], input_tensorL);
        pretreat->convert(imageR.data, w, h, imageR.step[0], input_tensorR);

        gettimeofday(&data_gpu_end,NULL);
        data_to_gpu=(data_gpu_end.tv_sec-data_gpu_start.tv_sec)*1000000+(data_gpu_end.tv_usec-data_gpu_start.tv_usec);
        std::cout<<"data_to_gpu time : "<<data_to_gpu/1000.0<<"ms"<<std::endl;


        float forward_time_use = 0;
        struct timeval forward_start;
        struct timeval forward_end;
        gettimeofday(&forward_start,NULL);

        my_interpreter->runSession(my_session);

        gettimeofday(&forward_end,NULL);
        forward_time_use=(forward_end.tv_sec-forward_start.tv_sec)*1000000+(forward_end.tv_usec-forward_start.tv_usec);
        std::cout<<"forward time : "<<forward_time_use/1000.0<<"ms"<<std::endl;

        float data_to_cpu = 0;
        struct timeval data_cpu_start;
        struct timeval data_cpu_end;
        gettimeofday(&data_cpu_start,NULL);

        auto output = my_interpreter->getSessionOutput(my_session, "output");
//        output = MNN::Express::_Reshape(output, {2, h, w});
        auto t_host = new MNN::Tensor(output, MNN::Tensor::CAFFE);

        output->copyToHostTensor(t_host);


        float *value = t_host->host<float>();

        gettimeofday(&data_cpu_end,NULL);
        data_to_cpu=(data_cpu_end.tv_sec-data_cpu_start.tv_sec)*1000000+(data_cpu_end.tv_usec-data_cpu_start.tv_usec);
        std::cout<<"data_to_cpu time : "<<data_to_cpu/1000.0<<"ms"<<std::endl;
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
        cv::Mat colorimg;
        cv::Mat colorimgfinal;
//        cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)
        cv::convertScaleAbs(showImg,colorimg);
        cv::applyColorMap(colorimg,colorimgfinal,cv::COLORMAP_PARULA);
        namedWindow("image", cv::WINDOW_AUTOSIZE);
        imshow("image", colorimgfinal);
        cv::waitKey(0);

        std::cout << "success" << std::endl;
    }




    return 0;
}
