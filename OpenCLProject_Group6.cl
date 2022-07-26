#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif


// Read value from global array a, return 0 if outside image
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

//Read value from global array a, return 0 if outside vector
float getValueGlobal(__global const float* a, size_t countX, size_t countY, int i, int j) {
	if (i < 0 || i >= countX || j < 0 || j >= countY)
		return 0;
	else
		return a[countX * j + i];
}

//Read value from global array b, return 0 if outside vector
int getValueMask(__global const int* b, size_t row, size_t column, int i, int j)
{
	if (i < 0 || i >= row || j < 0 || j >= column)
		return 0;
	else
		return b[j * row + i];

}

//Read value from the input image
float getValueImage(__read_only image2d_t val, int i, int j) 
{
	return read_imagef(val, sampler, (int2) { i, j }).x;
}

//Function to perform Sort of numbers in array
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


// Dilation Kernel
__kernel void dilationKernel(__read_only image2d_t d_input, __global float* d_outputDilation , __global int* d_structure_element) {
	
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float maximum = 0.0f;
	
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			//const float pxVal = read_imagef(d_input, sampler, (int2)(x+i, y+j)).x;
			const float pixelVal = getValueImage(d_input, x + i, y + j);
		
			const int mask_value = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
		    //printf("mask_value dilation %d\n", mask_value);
			if (mask_value == 1)
			{
				if (maximum > pixelVal)
					maximum = maximum;
				else
					maximum = pixelVal;
			}
		}
	}
	if (maximum <= 0.5)
		maximum = 0;
	else
		maximum = 1;

	d_outputDilation[countX * j + i] = maximum;
}

//Erosion Kernel
__kernel void erosionKernel(__read_only image2d_t d_input, __global float* d_outputErosion, __global int* d_structure_element) {
	
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);
	
	float minimum = 1.0f;

	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			const float pixelVal = getValueImage(d_input, x + i, y + j);
			int mask_value = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			if (mask_value == 1)
			{
				if (minimum < pixelVal)
					minimum = minimum;
				else
					minimum = pixelVal;
			}
		}
	}

	if (minimum <= 0.5)
		minimum = 0;
	else
		minimum = 1;

	d_outputErosion[countX * j + i] = minimum;
}

// Opening kernel
__kernel void openingKernel(__read_only image2d_t d_input, __global float* d_outputOpening, __global float* d_temp1, __global int* d_structure_element) {

	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float minimum = 1.0f;
	float maximum = 0.0f;
	
	// Erosion Operation
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			const float pixelVal1 = getValueImage(d_input, x + i, y + j);
			int mask_value = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			
			if (mask_value == 1)
			{
				if (minimum < pixelVal1)
					minimum = minimum;
				else
					minimum = pixelVal1;
			}
		}
	}
	d_temp1[countX * j + i] = minimum; // store the result of erosion in temporary array. It will act as input for dilation operation
	// global barrier
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Dilation Operation
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			
			const float pixelVal2 = getValueGlobal(d_temp1, countX, countY, x + i, y + j);
			int mask_value2 = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			
			if (mask_value2 == 1)
			{
				if (maximum > pixelVal2)
					maximum = maximum;
				else
					maximum = pixelVal2;
			}
		}
	}

	if (maximum <= 0.5)
		maximum = 0;
	else
		maximum = 1;

	d_outputOpening[countX * j + i] = maximum;
}

// Closing Kernel   
__kernel void closingKernel(__read_only image2d_t d_input, __global float* d_outputClosing, __global float* d_temp2, __global int* d_structure_element) {

	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	float minimum = 1.0f;
	float maximum = 0.0f;

	// Dilation Operation
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			const float pixelVal1 = getValueImage(d_input, x + i, y + j);
			int mask_value = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			
			if (mask_value == 1)
			{
				if (maximum > pixelVal1)
					maximum = maximum;
				else
					maximum = pixelVal1;
			}
		}
	}
	//store the result of dilation in temporary array. It will act as input for erosion
	d_temp2[countX * j + i] = maximum;
	barrier(CLK_GLOBAL_MEM_FENCE); // global barrier
	// Erosion Operation 
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			const float pixelVal2 = getValueGlobal(d_temp2, countX, countY, x + i, y + j);
			int mask_value2 = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			if (mask_value2 == 1)
			{
				if (minimum < pixelVal2)
					minimum = minimum;
				else
					minimum = pixelVal2;
			}
        }
	}
	if (minimum <= 0.5)
		minimum = 0;
	else
		minimum = 1;

	d_outputClosing[countX * j + i] = minimum;
}

// Kernel for Median filter dimension 3x3
__kernel void median1Kernel(__read_only image2d_t d_input, __global float* d_outputMedianGpu)  {
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	int msize = 3;
	int kernel_values_m3[9];

	for (int k = 0; k < 9; k++)
	{
		kernel_values_m3[k] = 0;
	}

	for (int x = i-1; x <= i+1; ++x)
	{
		for (int y = j-1; y <= j+1; ++y)
		{
			int x_1 = x - i + 1;
			int y_1 = y - j + 1;
			float pixelVal = getValueImage(d_input, x, y);
			//printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pixelVal);

			kernel_values_m3[x_1 + msize * y_1] = pixelVal;
		}
	}

	for (int s = 0; s < 9; s++)
	{
		if (kernel_values_m3[s] <= 0.5)
			kernel_values_m3[s] = 0;
		else
			kernel_values_m3[s] = 1;
	}

	sort(kernel_values_m3, 9);
	
	d_outputMedianGpu[j * countX + i] = kernel_values_m3[4];
}

// Kernel for Median filter dimension 5x5 
__kernel void median2Kernel(__read_only image2d_t d_input, __global float* d_outputMedianGpu) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);
	
	int msize = 5;
	int kernel_values_m5[25];
	int count;
	int temp;

	for (int k = 0; k < 25; k++)
	{
		kernel_values_m5[k] = 0;
	}

	for (int x = i - 2; x <= i + 2; ++x)
	{
		for (int y = j - 2; y <= j + 2; ++y)
		{
			int x_1 = x - i + 2;
			int y_1 = y - j + 2;
			float pixelVal = getValueImage(d_input, x, y);
			//printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pixelVal);

			kernel_values_m5[x_1 + msize * y_1] = pixelVal;

		}
	}

	for (int s = 0; s < 25; s++)
	{
		if (kernel_values_m5[s] <= 0.5)
			kernel_values_m5[s] = 0;
		else
			kernel_values_m5[s] = 1;
	}
	sort(kernel_values_m5, 25);

	d_outputMedianGpu[j * countX + i] = kernel_values_m5[12];
}