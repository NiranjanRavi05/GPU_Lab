//////////////////////////////////////////////////////////////////////////////
// OpenCL Lab Project: Group 6
// 1. Morphological filters: Dilation, Erosion, Opening and Closing
// 2. Median filter ( masks => 3x3 and 5x5 ) 
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>
#include <math.h>


#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>


//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////

const int msize1 = 3; // size of mask 1 
const int msize2 = 5; // size of mask 2

float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j) {
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[j * countX + i];
}
int getValueMask(const std::vector<int>& b, std::size_t row, std::size_t column, int i, int j)
{
	if (i < 0 || (size_t)i >= row || j < 0 || (size_t)j >= column)
		return 0;
	else
		return b[j * row + i];
}


bool compareResult(const std::vector<float>& hOutputCPU, const std::string& outputCPU, const std::vector<float>& hOutputGPU, const std::string& outputGPU, std::size_t countX, std::size_t countY) {
	std::size_t errorCount = 0;
	for (size_t j = 0; j < countY; j = j + 1) { //loop in the y-direction
		for (size_t i = 0; i < countX; i = i + 1) { //loop in the x-direction
			size_t index = i + j * countX;
			// Allow small differences between results (due to rounding)
			if (!(std::abs(hOutputCPU[index] - hOutputGPU[index]) <= 1e-2)) {
				if (errorCount < 15)
					std::cout << "Result for " << i << "," << j << " is incorrect: " << outputCPU << " value is " << hOutputCPU[index] << ", " << outputGPU << " value is " << hOutputGPU[index] << std::endl;
				else if (errorCount == 15)
					std::cout << "..." << std::endl;
				errorCount++;
			}
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return false;
	}
	return true;
}


// Function used for dilation operation
void dilation(const std::vector<float>& h_input, std::vector<int> const& h_structure_element,std::vector<float>& h_outputDilationCpu, std::size_t countX, std::size_t countY) 
{
	float maximum = 0.0f;
	std::size_t mask_size = 3;
	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			maximum = 0.0f;
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					float pixelValue = getValueGlobal(h_input, countX, countY, x + i, y + j);
					int mask_element = getValueMask(h_structure_element, mask_size, mask_size, x + 1, y + 1);
					if (mask_element == 1)
					{

						if (maximum > pixelValue)
							maximum = maximum;
						else
							maximum = pixelValue;
					}

				}
			}
			if (maximum <= 0.5)
				maximum = 0;
			else
				maximum = 1;

			h_outputDilationCpu[j * countX + i] = maximum;
		}
	}
}


// Function used for erosion operation
void erosion(const std::vector<float>& h_input, std::vector<int>& h_structure_element,std::vector<float>& h_outputErosionCpu, std::size_t countX, std::size_t countY) 
{
	float minimum = 1.0f;
	std::size_t mask_size = 3;
	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			minimum = 1.0f;
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					float pixelValue = getValueGlobal(h_input, countX, countY, x + i, y + j);
					int mask_element = getValueMask(h_structure_element, mask_size, mask_size, x + 1, y + 1);
					if (mask_element == 1)
					{

						if (minimum < pixelValue)
							minimum = minimum;
						else
							minimum = pixelValue;
					}

				}
			}
			
			if (minimum <= 0.5)
				minimum = 0;
			else
				minimum = 1;

			h_outputErosionCpu[j * countX + i] = minimum;
		}
	}
}


// Function used for opening operation
void opening(const std::vector<float>& h_input, std::vector<int>& h_structure_element, std::vector<float>& h_outputOpeningCpu, std::size_t countX, std::size_t countY)
{
	std::size_t count_opening = countX * countY;
	std::vector<float> h_temp_opening(count_opening);
	erosion(h_input, h_structure_element, h_temp_opening, countX, countY);
	dilation(h_temp_opening, h_structure_element, h_outputOpeningCpu, countX, countY);
}

// Function used for closing operation
void closing(const std::vector<float>& h_input, std::vector<int>& h_structure_element, std::vector<float>& h_outputClosingCpu, std::size_t countX, std::size_t countY)
{
	std::size_t count_closing = countX * countY;
	std::vector<float> h_temp_closing(count_closing);
	dilation(h_input, h_structure_element, h_temp_closing, countX, countY);
	erosion(h_temp_closing, h_structure_element, h_outputClosingCpu, countX, countY);
}

//Function to perform Sort
void sort(int array[], int size)
{
	int i, j, min;
	float temp;
	for (i = 0; i < size - 1; i++)
	{
		temp = 0;
		min = i;
		for (j = i + 1; j < size; j++)
			if (array[j] < array[min])
				min = j;
		temp = array[min];
		array[min] = array[i];
		array[i] = temp;

	}
}

// Function used for median_filter operation
void median_filter(const std::vector<float>& h_input,std::vector<float>& h_MedianOut,std::size_t countX,std::size_t countY,int msize)
{
	int kernel_values_1d[25];
	int count;
	int temp;

	float sum = 0.0;
	int a = (msize - 1) / 2;

	for (int i = 0; i < countX; ++i) {
		for (int j = 0; j < countY; ++j) {
			for (int x = i - a; x <= i + a; ++x) {
				for (int y = j - a; y <= j + a; ++y) {
					int x_1 = x - i + a;
					int y_1 = y - j + a;

					kernel_values_1d[x_1 + msize * y_1] = getValueGlobal(h_input, countX, countY, x, y);
				}
			}

			sort(kernel_values_1d, msize*msize);

			h_MedianOut[j * countX + i] = kernel_values_1d[a + msize * a];
		}
	}
}


//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	// Create a context
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);


	// Get the first device of the context
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "C:/Personal/Studies/SS2022/GPU Lab/Opencl-Basics-Windows/Opencl-ex1/src/OpenCLProject_Group6.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	std::vector<float> inputData;
	std::size_t inputWidth, inputHeight;
	Core::readImagePGM("C:/Personal/Studies/SS2022/GPU Lab/Opencl-Basics-Windows/Opencl-ex1/src/input_image.pgm", inputData, inputWidth, inputHeight);

	// Declare some values 
	std::size_t wgSizeX = 20; // Number of work items per work group in X direction
	std::size_t wgSizeY = 20;
	std::size_t countX = wgSizeX * std::size_t(inputWidth/wgSizeX); // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * std::size_t(inputHeight/wgSizeY);
	std::size_t count = countX * countY; // Overall number of elements
	// Number of elements in structuring element which is used for dilation, erosion, opening and closing
	std::size_t countA =(size_t)(msize1 * msize1);
	// Number of elements in mask of dimension 3x3
	std::size_t countB = (size_t)(msize1 * msize1);
	// Number of elements in mask of dimension 5x5
	std:size_t countD = (size_t)(msize2 * msize2);
	std::size_t size = count * sizeof(float); // Size of data in bytes
	std::size_t sizeA = countA * sizeof(int);
	std::size_t sizeB = countB * sizeof(float);
	std::size_t sizeD = countD * sizeof(float);
	
	// structuring element of size 3x3 for dilation, erosion, opening and closing (2D)
	std::vector<std::vector<int>> structure_element_2d{
				{ 0, 1, 0},
				{ 1, 1, 1},
				{ 0, 1, 0}
	};

	// structuring element of size 3x3 for dilation, erosion, opening and closing (1D)
	int structure_element_1d[msize1 * msize1];

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input(count);
	std::vector<float> h_outputDilationCpu(count);
	std::vector<float> h_outputErosionCpu(count);

	std::vector<float> h_outputOpeningCpu(count);
	std::vector<float> h_outputClosingCpu(count);

	std::vector<float> h_outputDilationGpu(count);
	std::vector<float> h_outputErosionGpu(count);

	std::vector<float> h_outputOpeningGpu(count);
	std::vector<float> h_outputClosingGpu(count);

	std::vector<float> h_temp1(count);
	std::vector<float> h_temp2(count);

	std::vector<float> h_outputMedian1Cpu(count);
	std::vector<float> h_outputMedian1Gpu(count);

	std::vector<float> h_outputMedian2Cpu(count);
	std::vector<float> h_outputMedian2Gpu(count);

	std::vector<int> h_structure_element(countA);

	// Allocate space for input and output data on the device
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_structure_element(context, CL_MEM_READ_WRITE, sizeA);
	cl::Buffer d_outputDilation(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputErosion(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputOpening(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputClosing(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_temp1(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_temp2(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputMedian1Gpu(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_outputMedian2Gpu(context, CL_MEM_READ_WRITE, size);
	
	
	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
    memset(h_structure_element.data(), 255, sizeA);

	memset(h_outputDilationCpu.data(), 255, size);
	memset(h_outputErosionCpu.data(), 255, size);

	memset(h_outputOpeningCpu.data(), 255, size);
	memset(h_outputClosingCpu.data(), 255, size);

	memset(h_outputDilationGpu.data(), 255, size);
	memset(h_outputErosionGpu.data(), 255, size);

	memset(h_outputOpeningGpu.data(), 255, size);
	memset(h_outputClosingGpu.data(), 255, size);

	memset(h_temp2.data(), 255, size);
	memset(h_temp1.data(), 255, size);

	memset(h_outputMedian1Cpu.data(), 255, size);
	memset(h_outputMedian1Gpu.data(), 255, size);

	memset(h_outputMedian2Cpu.data(), 255, size);
	memset(h_outputMedian2Gpu.data(), 255, size);

	//GPU
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_structure_element, true, 0, sizeA, h_structure_element.data());

	queue.enqueueWriteBuffer(d_outputDilation, true, 0, size, h_outputDilationGpu.data());
	queue.enqueueWriteBuffer(d_outputErosion, true, 0, size, h_outputErosionGpu.data());
	queue.enqueueWriteBuffer(d_outputOpening, true, 0, size, h_outputOpeningGpu.data());
	queue.enqueueWriteBuffer(d_outputClosing, true, 0, size, h_outputClosingGpu.data());

	queue.enqueueWriteBuffer(d_temp1, true, 0, size, h_temp1.data());
	queue.enqueueWriteBuffer(d_temp2, true, 0, size, h_temp2.data());

	queue.enqueueWriteBuffer(d_outputMedian1Gpu, true, 0, size, h_outputMedian1Gpu.data());
	queue.enqueueWriteBuffer(d_outputMedian2Gpu, true, 0, size, h_outputMedian2Gpu.data());

	//////// Load input data ////////////////////////////////
	for (size_t i = 0; i < countX; i++) {
		for (size_t j = 0; j < countY; j++) {
			h_input[j + countY * i] = inputData[(j % inputWidth) + inputWidth * (i % inputHeight)];
		}
	}
	
	// convert 2D structuring element into 1D structing element. 
    for (std::size_t i = 0; i < (countA)/3; i++)
	{
		for (std::size_t j = 0; j < (countA)/3; j++)
		{
			h_structure_element[i * ((countA)/3)+ j] = structure_element_2d[i][j];
		}
	}
	
	// Do calculation on the host side
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	dilation(h_input, h_structure_element, h_outputDilationCpu, countX, countY);
	erosion(h_input, h_structure_element, h_outputErosionCpu, countX, countY);
	opening(h_input, h_structure_element, h_outputOpeningCpu, countX, countY);
	closing(h_input, h_structure_element, h_outputClosingCpu, countX, countY);
	median_filter(h_input, h_outputMedian1Cpu, countX, countY, msize1);
	median_filter(h_input, h_outputMedian2Cpu, countX, countY, msize2);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();

	//////// Store CPU output image ///////////////////////////////////
	Core::writeImagePGM("output_dilation_cpu.pgm", h_outputDilationCpu, countX, countY);
	Core::writeImagePGM("output_erosion_cpu.pgm", h_outputErosionCpu, countX, countY);
	Core::writeImagePGM("output_opening_cpu.pgm", h_outputOpeningCpu, countX, countY);
	Core::writeImagePGM("output_closing_cpu.pgm", h_outputClosingCpu, countX, countY);
	Core::writeImagePGM("output_median1_cpu.pgm", h_outputMedian1Cpu, countX, countY);
	Core::writeImagePGM("output_median2_cpu.pgm", h_outputMedian2Cpu, countX, countY);
	Core::TimeSpan cpuExecTime = cpuEnd - cpuStart;
	std::cout << "Execution time on CPU is: " << cpuExecTime << std::endl;

	//Reinitialize output memory to 0xff
	memset(h_outputDilationGpu.data(), 255, size);
	memset(h_outputErosionGpu.data(), 255, size);
	memset(h_outputOpeningGpu.data(), 255, size);
	memset(h_outputClosingGpu.data(), 255, size);
	memset(h_outputMedian1Gpu.data(), 255, size);
	memset(h_outputMedian2Gpu.data(), 255, size);


	//GPU
	queue.enqueueWriteBuffer(d_outputDilation, true, 0, size, h_outputDilationGpu.data());
	queue.enqueueWriteBuffer(d_outputErosion, true, 0, size, h_outputErosionGpu.data());
	queue.enqueueWriteBuffer(d_outputOpening, true, 0, size, h_outputOpeningGpu.data());
	queue.enqueueWriteBuffer(d_outputClosing, true, 0, size, h_outputClosingGpu.data());
	queue.enqueueWriteBuffer(d_outputMedian1Gpu, true, 0, size, h_outputMedian1Gpu.data());
	queue.enqueueWriteBuffer(d_outputMedian2Gpu, true, 0, size, h_outputMedian2Gpu.data());

	// Copy input data to device
	cl::Event writeEvent[2];
	cl::Image2D image;

	image = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;
	cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;
	queue.enqueueWriteImage(image, true, origin, region, countX * sizeof(float), 0, h_input.data(), NULL, &writeEvent[0]);
	
	queue.enqueueWriteBuffer(d_structure_element, true, 0, sizeA, h_structure_element.data(), NULL, &writeEvent[1]);


	// Create a kernel object
	cl::Kernel dilationKernel(program, "dilationKernel");
	cl::Kernel erosionKernel(program, "erosionKernel");
	cl::Kernel openingKernel(program, "openingKernel");
	cl::Kernel closingKernel(program, "closingKernel");
	cl::Kernel median1Kernel(program, "median1Kernel");
	cl::Kernel median2Kernel(program, "median2Kernel");

	// Launch kernel on the device
	cl::Event kernelEvent[6];

	dilationKernel.setArg<cl::Image2D>(0, image);
	dilationKernel.setArg<cl::Buffer>(1, d_outputDilation);
	dilationKernel.setArg<cl::Buffer>(2, d_structure_element);
	queue.enqueueNDRangeKernel(dilationKernel, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &kernelEvent[0]);

	erosionKernel.setArg<cl::Image2D>(0, image);
	erosionKernel.setArg<cl::Buffer>(1, d_outputErosion);
	erosionKernel.setArg<cl::Buffer>(2, d_structure_element);
	queue.enqueueNDRangeKernel(erosionKernel, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &kernelEvent[1]);

	openingKernel.setArg<cl::Image2D>(0, image);
	openingKernel.setArg<cl::Buffer>(1, d_outputOpening);
	openingKernel.setArg<cl::Buffer>(2, d_temp1);
	openingKernel.setArg<cl::Buffer>(3, d_structure_element);
	queue.enqueueNDRangeKernel(openingKernel, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &kernelEvent[2]);

	closingKernel.setArg<cl::Image2D>(0, image);
	closingKernel.setArg<cl::Buffer>(1, d_outputClosing);
	closingKernel.setArg<cl::Buffer>(2, d_temp2);
	closingKernel.setArg<cl::Buffer>(3, d_structure_element);
	queue.enqueueNDRangeKernel(closingKernel, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &kernelEvent[3]);

	median1Kernel.setArg<cl::Image2D>(0, image);
	median1Kernel.setArg<cl::Buffer>(1, d_outputMedian1Gpu);
	queue.enqueueNDRangeKernel(median1Kernel, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &kernelEvent[4]);

	median2Kernel.setArg<cl::Image2D>(0, image);
	median2Kernel.setArg<cl::Buffer>(1, d_outputMedian2Gpu);
	queue.enqueueNDRangeKernel(median2Kernel, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &kernelEvent[5]);

	// Copy output data back to host
	cl::Event readEvent[6];
	queue.enqueueReadBuffer(d_outputDilation, true, 0, size, h_outputDilationGpu.data(), NULL, &readEvent[0]);
	queue.enqueueReadBuffer(d_outputErosion, true, 0, size, h_outputErosionGpu.data(), NULL, &readEvent[1]);
	queue.enqueueReadBuffer(d_outputOpening, true, 0, size, h_outputOpeningGpu.data(), NULL, &readEvent[2]);
	queue.enqueueReadBuffer(d_outputClosing, true, 0, size, h_outputClosingGpu.data(), NULL, &readEvent[3]);
	queue.enqueueReadBuffer(d_outputMedian1Gpu, true, 0, size, h_outputMedian1Gpu.data(), NULL, &readEvent[4]);
	queue.enqueueReadBuffer(d_outputMedian2Gpu, true, 0, size, h_outputMedian2Gpu.data(), NULL, &readEvent[5]);

	// Print performance data
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;

	Core::TimeSpan gpuTime = Core::TimeSpan::fromSeconds(0);
	for (std::size_t i = 0; i < sizeof(kernelEvent) / sizeof(*kernelEvent); i++)
		gpuTime = gpuTime + OpenCL::getElapsedTime(kernelEvent[i]);

	Core::TimeSpan copytime2 = Core::TimeSpan::fromSeconds(0);
	for (std::size_t i = 0; i < sizeof(readEvent) / sizeof(*readEvent); i++)
		copytime2 = copytime2 + OpenCL::getElapsedTime(readEvent[i]);

	Core::TimeSpan copytime1 = Core::TimeSpan::fromSeconds(0);
	for (std::size_t i = 0; i < sizeof(writeEvent) / sizeof(*writeEvent); i++)
		copytime1 = copytime1 + OpenCL::getElapsedTime(writeEvent[i]);

	Core::TimeSpan copyTime = copytime1 + copytime2;
	Core::TimeSpan overallGpuTime = gpuTime + copyTime;
	std::cout << "CPU Time: " << cpuTime.toString() << ", " << (count / cpuTime.getSeconds() / 1e6) << " MPixel/s" << std::endl;;
	std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
	std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ", " << (count / gpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;
	std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ", " << (count / overallGpuTime.getSeconds() / 1e6) << " MPixel/s)" << std::endl;

	//////// Store GPU output image ///////////////////////////////////
	Core::writeImagePGM("output_dilation_gpu.pgm", h_outputDilationGpu, countX, countY);
	Core::writeImagePGM("output_erosion_gpu.pgm", h_outputErosionGpu, countX, countY);
	Core::writeImagePGM("output_opening_gpu.pgm", h_outputOpeningGpu, countX, countY);
	Core::writeImagePGM("output_closing_gpu.pgm", h_outputClosingGpu, countX, countY);
	Core::writeImagePGM("output_median1_gpu.pgm", h_outputMedian1Gpu, countX, countY);
	Core::writeImagePGM("output_median2_gpu.pgm", h_outputMedian2Gpu, countX, countY);

	// compare the result of CPU and GPU
	if (!compareResult(h_outputDilationCpu, "DilationCPU", h_outputDilationGpu, "DilationGpu", countX, countY)) return 1;
	if (!compareResult(h_outputErosionCpu, "ErosionCPU", h_outputErosionGpu, "ErosionGpu", countX, countY)) return 1;
	if (!compareResult(h_outputOpeningCpu, "OpeningCPU", h_outputOpeningGpu, "OpeningGpu", countX, countY)) return 1;
	if (!compareResult(h_outputClosingCpu, "ClosingCPU", h_outputClosingGpu, "ClosingGpu", countX, countY)) return 1;
	if (!compareResult(h_outputMedian1Cpu, "Median1CPU", h_outputMedian1Gpu, "Median1Gpu", countX, countY)) return 1;
	if (!compareResult(h_outputMedian2Cpu, "Median2CPU", h_outputMedian2Gpu, "Median2Gpu", countX, countY)) return 1;

	std::cout << "Success" << std::endl;
	
	return 0;
}