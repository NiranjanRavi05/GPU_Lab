#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif


// Read value from global array a, return 0 if outside image
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
float getValueImage(__read_only image2d_t a, int i, int j) {
	 return read_imagef(a, sampler, (int2) { i, j }).x;
}

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
			const float pxVal = getValueImage(d_input, x + i, y + j);
		
			const int mask_value = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
		    //printf("mask_value dilation %d\n", mask_value);
			if (mask_value == 1)
			{
				if (maximum > pxVal)
					maximum = maximum;
				else
					maximum = pxVal;
			}


		}
		
	}
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
			//const float pxVal = read_imagef(d_input, sampler, (int2)(x+i, y+j)).x;
			const float pxVal = getValueImage(d_input, x + i, y + j);
			int mask_value = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			//printf("mask_value erosion %d\n", mask_value);
			//	printf("pixel value for co-ordinates %d, %d is %f\n", x + i, y + j, pxVal);
			if (mask_value == 1)
			{
				if (minimum < pxVal)
					minimum = minimum;
				else
					minimum = pxVal;
			}


		}
	}
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
			const float pxVal = getValueImage(d_input, x + i, y + j);
			int mask_value = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			
			if (mask_value == 1)
			{
				if (minimum < pxVal)
					minimum = minimum;
				else
					minimum = pxVal;
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
			
			const float pxVal2 = getValueGlobal(d_temp1, countX, countY, x + i, y + j);
			int mask_value2 = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			
			if (mask_value2 == 1)
			{
				if (maximum > pxVal2)
					maximum = maximum;
				else
					maximum = pxVal2;
			}


		}
	}
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
			const float pxVal2 = getValueImage(d_input, x + i, y + j);
			int mask_value = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			
			if (mask_value == 1)
			{
				if (maximum > pxVal2)
					maximum = maximum;
				else
					maximum = pxVal2;
			}


		}
	}
	//store the result of dilation in temporary array. It will act as input for erosion operation
	d_temp2[countX * j + i] = maximum;
	barrier(CLK_GLOBAL_MEM_FENCE); // global barrier
	// Erosion Operation 
	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			const float pxVal = getValueGlobal(d_temp2, countX, countY, x + i, y + j);
			int mask_value2 = getValueMask(d_structure_element, 3, 3, x + 1, y + 1);
			if (mask_value2 == 1)
			{
				if (minimum < pxVal)
					minimum = minimum;
				else
					minimum = pxVal;
			}
         }
	}
	d_outputClosing[countX * j + i] = minimum;

}

// Kernel for Gaussian filter (Gaussian mask dimension 3x3) 
__kernel void gaussian1(__read_only image2d_t d_input, __global float* d_outputGaussianGpu, __global float* d_gaussianMask)  {
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);
	
	float sum = 0.0; 
	int msize = 3;
	
	for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
				{
					float pixelValue = getValueImage(d_input, x + i, y + j);
					sum += (pixelValue) * getValueGlobal(d_gaussianMask, (size_t)msize, (size_t)msize, x+1,y+1) ;
				}
			}
	d_outputGaussianGpu[j * countX + i] = sum;
			
}
// Kernel for Gaussian filter (Gaussian mask dimension 5x5) 
__kernel void gaussian2(__read_only image2d_t d_input, __global float* d_outputGaussianGpu, __global float* d_gaussianMask) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);
	
	float sum = 0.0;
	int msize = 5;
	
	for (int x = -2; x <= 2; x++)
	{
		for (int y = -2; y <= 2; y++)
		{
			
			float pixelValue = getValueImage(d_input, x + i, y + j);
			sum += (pixelValue)*getValueGlobal(d_gaussianMask, (size_t)msize, (size_t)msize, x + 2, y + 2);
			
         }
	}
	d_outputGaussianGpu[j * countX + i] = sum;
	
}