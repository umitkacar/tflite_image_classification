#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <cmath>

#define LOG(severity) (std::cerr << (#severity) << ": ")

using namespace cv;

void test() {

		// Load model
		std::unique_ptr<tflite::FlatBufferModel> model;
		std::unique_ptr<tflite::Interpreter> interpreter;
		model = tflite::FlatBufferModel::BuildFromFile("../models/model.tflite");
		if (!model) {
    		LOG(ERROR) << "Not uploading model. " << "\n";
		}
		else
			LOG(INFO) << "Model is uploaded.... " << "\n";

		// Build the interpreter
		tflite::ops::builtin::BuiltinOpResolver resolver;
		tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
		if (!interpreter) {
    		LOG(ERROR) << "Failed to construct interpreter\n";
		}
		else {
			LOG(INFO) << "Model is constructed interpreter...\n";
		}

		interpreter->SetAllowFp16PrecisionForFp32(true);
		interpreter->SetNumThreads(4);

		// INFORMATION ABOUT MODEL
		// LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
		// LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
		// LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
		// LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

		// int t_size = interpreter->tensors_size();
		// for (int i = 0; i < t_size; i++) {
		// if (interpreter->tensor(i)->name)
		// 	LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
		// 			<< interpreter->tensor(i)->bytes << ", "
		// 			<< interpreter->tensor(i)->type << ", "
		// 			<< interpreter->tensor(i)->params.scale << ", "
		// 			<< interpreter->tensor(i)->params.zero_point << "\n";
		// }
  

		TfLiteTensor* output_classes = nullptr;

		auto cam = cv::VideoCapture(0);
		// auto cam = cv::VideoCapture("../demo.mp4");

		std::string file_name = "../models/labels.txt";

		std::ifstream file(file_name);
		if (!file) {
    		LOG(ERROR) << "Labels file " << file_name << " not found\n";
  		}
		else {
			LOG(INFO) << "Labels file " << file_name << " found\n";
		}

		std::vector<std::string> labels;
		labels.clear();
		std::string line;
		size_t found_label_count;
		int count = 0;
		while (std::getline(file, line)) {
			labels.push_back(line);
			count = count + 1;
			LOG(INFO) << "Label-" <<  count << " : " << line << "\n";
		}
		found_label_count = labels.size();
		LOG(INFO) << "Labels Size : " << found_label_count << "\n";

		auto cam_width = cam.get(cv::CAP_PROP_FRAME_WIDTH);
		auto cam_height = cam.get(cv::CAP_PROP_FRAME_HEIGHT);
		while (true) {
			cv::Mat image0;
			auto success = cam.read(image0);
			if (!success) {
				std::cout << "cam fail" << std::endl;
				break;
			}
			cv::Mat image;
			resize(image0, image, Size(224,224));

			// feed input
			int image_height = 224;
			int image_width = 224;
			int image_channels = 3;
			int number_of_pixels = image_height * image_width * image_channels;
			int base_index = 0;

			int input = interpreter->inputs()[0];
			LOG(INFO) << "input: " << input << "\n";





			if (interpreter->AllocateTensors() != kTfLiteOk) {
    			LOG(ERROR) << "Failed to allocate tensors!\n";
			}
			else {
				LOG(INFO) << "Allocate tensors!\n";
			}





			cv::putText(image0, "img", cv::Point(20, 20),
			cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 255, 255));
			cv::imshow("cam", image0);
			// cv::waitKey(25);
			auto k = cv::waitKey(25);
			if (k == 27) {
				break;}
		}
		cam.release();
		cv::destroyAllWindows();
}
int main(int argc, char** argv) {
    test();
    return 0;
}

// mkdir build && cd build && cmake .. && make