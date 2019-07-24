#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define DIMENSION "DIMENSION:"
#define ITEM_QTY "NUMBER OF ITEMS:"
#define KNAPSACK_CAPACITY "CAPACITY OF KNAPSACK:"
#define MIN_SPEED "MIN SPEED:"
#define MAX_SPEED "MAX SPEED:"
#define RENTING_RATIO "RENTING RATIO:"
#define EDGE_WEIGHT_TYPE "EDGE_WEIGHT_TYPE:"
#define NODE_COORD_SECTION "NODE_COORD_SECTION	(INDEX, X, Y):"
#define ITEMS_SECTION "ITEMS SECTION	(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):"

double Dimension, ItemQuantity, KnapsackCapacity, MinSpeed, MaxSpeed, RentingRatio;
char EdgeWeightType[1000];

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int FindCharacterPosition(char stringToSearch[], char characterToFind)
{
	int stringLength = 0, i, characterPosition = 0;

	stringLength = strlen(stringToSearch);
	for (i = 0; i < stringLength; i++)
	{
		if (stringToSearch[i] == characterToFind)
			characterPosition = i + 1;
	}

	return (characterPosition);
}

void SubString(char originalString[], char subString[], int position, int length)
{
	int c = 0, d = 0;
	char tempSubString[1000];

	while (c < length) {
		subString[c] = originalString[position + c - 1];
		c++;
	}

	subString[c] = '\0';
	c = 0;

	while (subString[c] != '\0')
	{
		if (subString[c] == ' ' || subString[c] == '	') {
			int temp = c + 1;
			if (subString[temp] != '\0') {
				while ((subString[temp] == ' ' || subString[c] == '	') && subString[temp] != '\0') {
					if (subString[temp] == ' ' || subString[c] == '	')
						c++;
					temp++;
				}
			}
		}
		tempSubString[d] = subString[c];
		c++;
		d++;
	}

	tempSubString[d] = '\0';
	strcpy(subString, tempSubString);
}

int CountFileLines(char fileName[]) {

	FILE *filePtr;
	int lineCount = 0;
	char chr;

	filePtr = fopen(fileName, "r");
	chr = getc(filePtr);
	while (chr != EOF)
	{
		if (chr == '\n')
			lineCount++;
		chr = getc(filePtr);
	}
	fclose(filePtr);
	return lineCount;
}

int CountMatrixRows(char fileName[], char sectionName[])
{
	FILE *filePtr;
	char str[255], sub[255];
	int lineCount = 0, initialPosition = 0, rows = 0;

	filePtr = fopen(fileName, "r");

	while (fgets(str, 100, filePtr) != NULL) {
		if (strncmp(str, sectionName, strlen(sectionName)) == 0) {
			initialPosition = lineCount;
		}
		SubString(str, sub, 1, 1);
		if (initialPosition != NULL && lineCount > initialPosition && isdigit(sub[0])) {
			rows++;
		}
		else if(initialPosition != NULL && lineCount > initialPosition && isalpha(sub[0]))
		{
			break;
		}
		lineCount++;
	}
	fclose(filePtr);
	return rows;
}

int** ExtractMatrix(char fileName[], char sectionName[], int col)
{
	FILE *filePtr;
	int** matrixResult;
	char str[255], sub[255];
	int lineCount = 0, initialPosition, rows;

	filePtr = fopen(fileName, "r");
	rows = CountMatrixRows(fileName, sectionName);
	
	matrixResult = new int*[rows];

	for (int h = 0; h < rows; h++) {
		matrixResult[h] = new int[col];
	}

	while (fgets(str, 60, filePtr) != NULL) {
		if (strncmp(str, sectionName, strlen(sectionName)) == 0) {
			initialPosition = lineCount;
		}
		SubString(str, sub, 0, 1);
		if (initialPosition != NULL && lineCount > initialPosition && isdigit(sub[0])) {
			rows++;
		}
		lineCount++;
	}

	return matrixResult;
}

int main()
{
	char file_name[255], str[255], sub[255];
	FILE *fp;
	int position;

	printf("Enter name of a file you wish to see\n");
	gets_s(file_name);

	fp = fopen(file_name, "r"); //read mode

	if (fp == NULL)
	{
		perror("Error while opening the file. \n");
		exit(EXIT_FAILURE);
	}
	
	printf("The contents of %s file are: \n", file_name);

	printf("The line quantity in file are: %d \n", CountFileLines(file_name));

	while (fgets(str, 60, fp) != NULL) {
		position = FindCharacterPosition(str, ':');
		//printf("Position of ':' \"%d\"\n", position);
		if (strncmp(str, DIMENSION, strlen(DIMENSION)) == 0)
		{
			SubString(str, sub, position + 1, strlen(str) - position);
			Dimension = atof(sub);
			printf("Dimension is \"%lf\"\n", Dimension);
		}
		else if (strncmp(str, ITEM_QTY, strlen(ITEM_QTY)) == 0)
		{
			SubString(str, sub, position + 1, strlen(str) - position);
			ItemQuantity = atof(sub);
			printf("Item Quantity is \"%lf\"\n", ItemQuantity);
		}
		else if (strncmp(str, KNAPSACK_CAPACITY, strlen(KNAPSACK_CAPACITY)) == 0)
		{
			SubString(str, sub, position + 1, strlen(str) - position);
			KnapsackCapacity = atof(sub);
			printf("Knapsack Capacity is \"%lf\"\n", KnapsackCapacity);
		}
		else if (strncmp(str, MIN_SPEED, strlen(MIN_SPEED)) == 0)
		{
			SubString(str, sub, position + 1, strlen(str) - position);
			MinSpeed = atof(sub);
			printf("Min Speed is \"%lf\"\n", MinSpeed);
		}
		else if (strncmp(str, MAX_SPEED, strlen(MAX_SPEED)) == 0)
		{
			SubString(str, sub, position + 1, strlen(str) - position);
			MaxSpeed = atof(sub);
			printf("Max Speed is \"%lf\"\n", MaxSpeed);
		}
		else if (strncmp(str, RENTING_RATIO, strlen(RENTING_RATIO)) == 0)
		{
			SubString(str, sub, position + 1, strlen(str) - position);
			//printf("Required substring is \"%s\"\n", sub);
			RentingRatio = atof(sub);
			printf("Renting Ratio is \"%lf\"\n", RentingRatio);
		}
		else if (strncmp(str, EDGE_WEIGHT_TYPE, strlen(EDGE_WEIGHT_TYPE)) == 0)
		{
			SubString(str, sub, position + 1, strlen(str) - position);			
			strcpy(EdgeWeightType, sub);
			printf("Edge Weight Type is \"%s\"\n", EdgeWeightType);
		}
			// 
			//puts(str);
	}

	printf("Matrix of Nodes has %d rows \n", CountMatrixRows(file_name, NODE_COORD_SECTION));
	printf("Matrix of items has %d rows \n", CountMatrixRows(file_name, ITEMS_SECTION));

	fclose(fp);
	return 0;
    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;*/
}

int FileExists(const char *path)
{
	// Try to open file
	FILE *fptr = fopen(path, "r");

	// If file doesn't exists
	if (fptr == NULL)
		return 0;

	// File exists hence close file and return true
	fclose(fptr);

	return 1;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
