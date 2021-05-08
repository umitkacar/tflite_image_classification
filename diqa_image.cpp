#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"

#include <glob.h>
#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <vector>
#include <string>

std::vector<std::string> globVector(const std::string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
    std::vector<std::string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        files.push_back(std::string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

bool diqa_test_image() {

	// Load model
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("../models/diqa_beta.tflite");
	if (!model) {
		std::cerr << "Failed to load model: " << std::endl;
		return false;
	}
	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
	if (!interpreter) {
		std::cerr << "Failed to create interpreter!" << std::endl;
		return false;
	}
	// Resize input tensors, if desired.

	TfLiteTensor* input_tensor_ = nullptr;
	TfLiteTensor* output_classes = nullptr;

	std::vector<std::string> labels;

	auto file_name="../models/diqa_beta_labels.txt";
	std::ifstream label_name(file_name);
	for(std::string line; getline(label_name, line);)
	{
		labels.push_back(line);

	}
	std::cout << "label-0 :" << labels[0] << std::endl;
	std::cout << "label-1 :" << labels[1] << std::endl;
	std::cout << "label-2 :" << labels[2] << std::endl;

	std::string image_path = "/media/umit/MoreData/xTensorFlowLite/datasets/ID_cards/NORMAL-NEW-ID-FRONT-HOLOGRAM/*";
	std::vector<std::string> files = globVector(image_path);

    // for (int ii = 0; ii < files.size(); ii++){
	for (int ii = 0; ii < files.size(); ii++){

	    // Path name
		std::cout << ii << ".images path = "<< files[ii] << std::endl;

		cv::Mat image0 = cv::imread(files[ii],cv::IMREAD_COLOR);
		if (image0.empty()){
			std::cerr << "Image not found" << std::endl;
			return false;
		} else {
			std::cout << "Image uploaded." << std::endl;
		}
		cv::Mat image;
		cv::resize(image0, image, cv::Size(224,224));
		// cv::subtract(sample_float, cv::Scalar(B_MEAN, G_MEAN, R_MEAN), sample_float);
		image.convertTo(image, CV_32FC3, 1.f / 255);

		// std::cout << "check_image_type = " << image.type() <<std::endl;
        // for (int jj = 0; jj< 10; jj++){

		// 	cv::Vec3b pixel = image.at<cv::Vec3b>(jj, jj);
		// 	int blue = pixel.val[0];
		// 	int green = pixel.val[1];
		// 	int red = pixel.val[2];
		// 	std::cout << "blue = " << blue <<std::endl;
		// 	std::cout << "green = " << green <<std::endl;
		// 	std::cout << "red = " << red <<std::endl;

		// }

		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

		interpreter->AllocateTensors();

		const int input = interpreter->inputs()[0];
		TfLiteTensor* input_tensor = interpreter->tensor(input);
		TfLiteIntArray* input_dims = input_tensor->dims;
		const int width = input_dims->data[1];
		const int height = input_dims->data[2];
		const int channel = input_dims->data[3];
		const int size = input_dims->data[1] * input_dims->data[2] * input_dims->data[3];

		// std::cout << "width :" << input_dims->data[1]  << std::endl;
		// std::cout << "height :" << input_dims->data[2]  << std::endl;
		// std::cout << "channel :" << input_dims->data[3]  << std::endl;
		// std::cout << "input_size :" << input_dims->data[1] * input_dims->data[2] * input_dims->data[3]  << std::endl;
		// std::cout << "imgMat width*height :" << image.total()  << std::endl;
		// std::cout << "imgMat channel :" << image.elemSize()  << std::endl;

        float *dst = input_tensor->data.f;
   	    memcpy(dst, image.data, sizeof(float) * input_dims->data[1] * input_dims->data[2] * input_dims->data[3]);
     
		// interpreter->SetAllowFp16PrecisionForFp32(true);
		// interpreter->SetNumThreads(4);

		if (interpreter->Invoke() != kTfLiteOk){
			std::cerr <<  "Invoke Problem !";
			return false;
		}

		std::cout << "output_size :" << interpreter->outputs().size() << std::endl;
		output_classes = interpreter->tensor(interpreter->outputs()[0]);
		auto out_cls = output_classes->data.f;

		std::vector<float> cls;
		for (int j = 0; j < 3; j++){
			std::cout << "cls = "<< out_cls[j] << std::endl;
			cls.push_back(out_cls[j]);
		}

		cv::putText(image0, labels[0]+ " = " + std::to_string(cls[0]), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255),2);
		cv::putText(image0, labels[1]+ " = " + std::to_string(cls[1]), cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255),2);
		cv::putText(image0, labels[2]+ " = " + std::to_string(cls[2]), cv::Point(10, 90), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 0),2);

		cv::resize(image0, image0, cv::Size(), 1, 1);
		const std::string window_name{"image_test"};
		cv::namedWindow(window_name);
		cv::imshow(window_name, image0);
		cv::waitKey(5);
    }

}
int main(int argc, char** argv) {

    diqa_test_image();

    return 0;
}
