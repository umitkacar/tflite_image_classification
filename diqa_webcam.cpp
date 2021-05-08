#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <cmath>

bool diqa_test_video() {

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

		auto cam = cv::VideoCapture(0);
		// auto cam = cv::VideoCapture("../demo.mp4");

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

		auto cam_width = cam.get(cv::CAP_PROP_FRAME_WIDTH);
		auto cam_height = cam.get(cv::CAP_PROP_FRAME_HEIGHT);
		while (true) {
			cv::Mat image0;
			auto success = cam.read(image0);
			if (!success) {
				std::cout << "failing camera !!!" << std::endl;
				break;
			}
			cv::Mat image;
			cv::resize(image0, image, cv::Size(224,224));
			cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
			// image.convertTo(image, CV_32F, 1.f/255);
			interpreter->AllocateTensors();

			const int input = interpreter->inputs()[0];
			TfLiteTensor* input_tensor = interpreter->tensor(input);
			TfLiteIntArray* input_dims = input_tensor->dims;
			const int width = input_dims->data[1];
			const int height = input_dims->data[2];
			const int channel = input_dims->data[3];
			const int size = input_dims->data[1] * input_dims->data[2] * input_dims->data[3];

			std::cout << "width :" << input_dims->data[1]  << std::endl;
			std::cout << "height :" << input_dims->data[2]  << std::endl;
			std::cout << "channel :" << input_dims->data[3]  << std::endl;
			std::cout << "input_size :" << input_dims->data[1] * input_dims->data[2] * input_dims->data[3]  << std::endl;
			std::cout << "imgMat width*height :" << image.total()  << std::endl;
			std::cout << "imgMat channel :" << image.elemSize()  << std::endl;

			const int row_elems = width * channel;
			for (int row = 0; row < height; row++) {
				const uchar* row_ptr = image.ptr(row);
				for (int i = 0; i < row_elems; i++) {
					input_tensor->data.f[i + row_elems*row] = (row_ptr[i] - 0) / 255;
				}
			}

			// // copy image to input as input tensor
			// memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());
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
			cv::imshow("cam", image0);
			cv::waitKey(25);
			auto k = cv::waitKey(5);
			if (k == 27) {
				break;}
		}
		cam.release();
		cv::destroyAllWindows();
}
int main(int argc, char** argv) {
    diqa_test_video();
    return 0;
}
