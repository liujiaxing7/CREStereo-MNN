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

    const std::string mnn_path = argv[1];
    std::shared_ptr<MNN::Interpreter> my_interpreter = std::shared_ptr<MNN::Interpreter>(
            MNN::Interpreter::createFromFile(mnn_path.c_str()));

    // config
    MNN::ScheduleConfig config;
    int num_thread = 4;
    config.numThread = num_thread;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    config.backendConfig = &backendConfig;
    int forward = MNN_FORWARD_OPENCL;
    config.type = static_cast<MNNForwardType>(forward);

    // create session
    MNN::Session *my_session = my_interpreter->createSession(config);
    my_interpreter->releaseModel();

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

        std::shared_ptr<MNN::CV::ImageProcess> pretreat(
        MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::RGB));
        pretreat->convert(imageL.data, w, h, imageL.step[0], input_tensorL);
        pretreat->convert(imageR.data, w, h, imageR.step[0], input_tensorR);

        float forward_time_use = 0;
        struct timeval forward_start;
        struct timeval forward_end;
        gettimeofday(&forward_start,NULL);

        my_interpreter->runSession(my_session);

        auto output = my_interpreter->getSessionOutput(my_session, "output");
        auto t_host = new MNN::Tensor(output, MNN::Tensor::CAFFE);
        output->copyToHostTensor(t_host);

        gettimeofday(&forward_end,NULL);
        forward_time_use=(forward_end.tv_sec-forward_start.tv_sec)*1000000+(forward_end.tv_usec-forward_start.tv_usec);
        std::cout<<"forward time : "<<forward_time_use<<std::endl;

        float *output_array_boxes = t_host->host<float>();

//        std::vector<float> output_vector_boxes{output_array_boxes, output_array_boxes + 400};
//        std::cout<<"get output"<<std::endl;
//        for (int i = 0; i<400; ++i)
//        {
//            std::cout<<output_vector_boxes.at(i)<<std::endl;
//        }
//        std::cout<<"output:"<<output_array_boxes<<std::endl;
    }




    return 0;
}
