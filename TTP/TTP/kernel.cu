#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <vector>
#include <sstream>

#define DIMENSION "DIMENSION:"
#define ITEM_QTY "NUMBER OF ITEMS:"
#define KNAPSACK_CAPACITY "CAPACITY OF KNAPSACK:"
#define MIN_SPEED "MIN SPEED:"
#define MAX_SPEED "MAX SPEED:"
#define RENTING_RATIO "RENTING RATIO:"
#define EDGE_WEIGHT_TYPE "EDGE_WEIGHT_TYPE:"
#define NODE_COORD_SECTION "NODE_COORD_SECTION	(INDEX, X, Y):"
#define ITEMS_SECTION "ITEMS SECTION	(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):"

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

void cudaCheckError()
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		fprintf(stderr, "CUDA Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
	}
}

size_t findCharacterPosition(char stringToSearch[], char characterToFind)
{
	size_t stringLength = 0, i, characterPosition = 0;

	stringLength = strlen(stringToSearch);
	for (i = 0; i < stringLength; i++)
	{
		if (stringToSearch[i] == characterToFind)
			characterPosition = i + 1;
	}

	return (characterPosition);
}

void subString(char originalString[], char subString[], size_t position, size_t length)
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

int countFileLines(char fileName[]) {

	FILE* filePtr;
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

int countMatrixRows(const char fileName[], const char sectionName[])
{
	FILE* filePtr;
	char str[255], sub[255];
	int lineCount = 0, initialPosition = 0, rows = 0;
	const char openMode[] = "r";

	filePtr = fopen(fileName, openMode);

	while (fgets(str, 100, filePtr) != NULL) {
		if (strncmp(str, sectionName, strlen(sectionName)) == 0) {
			initialPosition = lineCount;
		}
		subString(str, sub, 1, 1);
		if (initialPosition != NULL && lineCount > initialPosition && isdigit(sub[0])) {
			rows++;
		}
		else if (initialPosition != NULL && lineCount > initialPosition && isalpha(sub[0]))
		{
			break;
		}
		lineCount++;
	}
	fclose(filePtr);
	return rows;
}

std::vector<std::string> split(const std::string& s, char delimiter)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);

	while (std::getline(tokenStream, token, delimiter))
	{
		tokens.push_back(token);
	}
	return tokens;
}

int fileExists(const char* path)
{
	// Try to open file
	FILE* fptr = fopen(path, "r");

	// If file doesn't exists
	if (fptr == NULL)
		return 0;

	// File exists hence close file and return true
	fclose(fptr);

	return 1;
}

int** extractMatrix(const char fileName[], const char sectionName[], int col)
{
	FILE* filePtr;
	char str[255], sub[255], * token;
	int lineCount = 0, initialPosition = 0, rows, matrixRow, matrixCol;
	const char openMode[] = "r";

	filePtr = fopen(fileName, openMode);
	rows = countMatrixRows(fileName, sectionName);

	int** matrixResult = (int**)malloc(rows * sizeof(int));
	for (int i = 0; i < col; i++) {
		matrixResult[i] = (int*)malloc(sizeof(int) * col);
	}

	while (fgets(str, 100, filePtr) != NULL) {
		if (strncmp(str, sectionName, strlen(sectionName)) == 0) {
			initialPosition = lineCount;
		}
		subString(str, sub, 1, 1);
		if (initialPosition != NULL && lineCount > initialPosition && isdigit(sub[0])) {
			token = strtok(str, "	");
			matrixCol = 0;
			matrixRow = atoi(token) - 1;
			while (token != NULL)
			{
				matrixResult[matrixRow][matrixCol] = atoi(token);
				token = strtok(NULL, "	");
				if (matrixCol < col)
					matrixCol++;
			}
		}
		else if (initialPosition != NULL && lineCount > initialPosition && isalpha(sub[0]))
		{
			break;
		}
		lineCount++;
	}

	fclose(filePtr);

	return matrixResult;
}

int** extractMatrix(const char fileName[], const char sectionName[], int rows, int cols)
{
	FILE* filePtr;
	char str[255], sub[255], * token;
	int lineCount = 0, initialPosition = 0, matrixRow, matrixCol;
	const char openMode[] = "r";

	filePtr = fopen(fileName, openMode);

	// Allocate memory for rows
	int **matrixResult = (int**)malloc(rows * sizeof(int*));
	if (matrixResult == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	// Allocate memory for columns
	for (int i = 0; i < rows; i++) {
		matrixResult[i] = (int*)malloc(cols * sizeof(int));
		if (matrixResult[i] == NULL) {
			fprintf(stderr, "Out of Memory");
			exit(0);
		}
	}
	
	while (fgets(str, 100, filePtr) != NULL) {
		if (strncmp(str, sectionName, strlen(sectionName)) == 0) {
			initialPosition = lineCount;
		}
		subString(str, sub, 1, 1);
		if (initialPosition != NULL && lineCount > initialPosition && isdigit(sub[0])) {
			token = strtok(str, "	");
			matrixCol = 0;
			matrixRow = atoi(token) - 1;
			while (token != NULL)
			{
				matrixResult[matrixRow][matrixCol] = atoi(token);
				token = strtok(NULL, "	");
				if (matrixCol < cols)
					matrixCol++;
			}
		}
		else if (initialPosition != NULL && lineCount > initialPosition && isalpha(sub[0]))
		{
			break;
		}
		lineCount++;
	}

	fclose(filePtr);

	return matrixResult;
}

void display(int** matrix, int rows, int columns) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%d ", matrix[i][j]);			
		}
		printf("\n");
	}
	printf("\n");
}

void display(float** matrix, int rows, int columns) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void euclideanDistance(int** srcPoint, int** dstPoint, float** out, int rCount, int size) {
	for (int s = 0; s < size; s++) {
		for (int xSrc = 0; xSrc < rCount; xSrc++) {
			for (int xDst = 0; xDst < rCount; xDst++) {
				out[s][0] = (float)srcPoint[xSrc][0];
				out[s][1] = (float)dstPoint[xDst][0];
				out[s][2] = (float)sqrt(pow(dstPoint[xDst][1] - srcPoint[xSrc][1], 2) + pow(dstPoint[xDst][2] - srcPoint[xSrc][2], 2) * 1.0);
				s++;
			}
		}
	}
}

__global__ void euclideanDistanceParallel(int** srcPts, int** dstPts, float** outDistMat, int rQty, int size) {
	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;
	for (int s = 0; s < size; s++) {
		for (int xSrc = 0; xSrc < rQty; xSrc++) {
			for (int xDst = 0; xDst < rQty; xDst++) {
				outDistMat[s][0] = srcPts[xSrc][0];
				outDistMat[s][1] = dstPts[xDst][0];
				outDistMat[s][2] = sqrt(pow(dstPts[xDst][1] - srcPts[xSrc][1], 2) + pow(dstPts[xDst][2] - srcPts[xSrc][2], 2) * 1.0);
				s++;
			}
		}
	}
}

int main()
{
	char file_name[255], str[255], sub[255];
	FILE* fp;
	size_t position;
	int** nodeMatrix;
	int** itemMatrix;
	const char openMode[] = "r";
	double Dimension, ItemQuantity, KnapsackCapacity, MinSpeed, MaxSpeed, RentingRatio;
	char EdgeWeightType[1000];

	// Ask for the filepath & name where the problem is defined
	printf("Enter name of a file you wish to see\n");
	gets_s(file_name);

	// Open the file in read mode
	fp = fopen(file_name, openMode);

	// Valida que no se presente algun error en la apertura del archivo
	if (fp == NULL)
	{
		perror("Error while opening the file. \n");
		exit(EXIT_FAILURE);
	}

	// Print headers
	printf("The contents of %s file are: \n", file_name);

	printf("The line quantity in file are: %d \n", countFileLines(file_name));

	// Obtain general data from file
	while (fgets(str, 100, fp) != NULL) {
		position = findCharacterPosition(str, ':');
		if (strncmp(str, DIMENSION, strlen(DIMENSION)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			Dimension = atof(sub);
			printf("Dimension is %lf \n", Dimension);
		}
		else if (strncmp(str, ITEM_QTY, strlen(ITEM_QTY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			ItemQuantity = atof(sub);
			printf("Item Quantity is %lf \n", ItemQuantity);
		}
		else if (strncmp(str, KNAPSACK_CAPACITY, strlen(KNAPSACK_CAPACITY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			KnapsackCapacity = atof(sub);
			printf("Knapsack Capacity is %lf \n", KnapsackCapacity);
		}
		else if (strncmp(str, MIN_SPEED, strlen(MIN_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			MinSpeed = atof(sub);
			printf("Min Speed is %lf \n", MinSpeed);
		}
		else if (strncmp(str, MAX_SPEED, strlen(MAX_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			MaxSpeed = atof(sub);
			printf("Max Speed is %lf \n", MaxSpeed);
		}
		else if (strncmp(str, RENTING_RATIO, strlen(RENTING_RATIO)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			RentingRatio = atof(sub);
			printf("Renting Ratio is %lf \n", RentingRatio);
		}
		else if (strncmp(str, EDGE_WEIGHT_TYPE, strlen(EDGE_WEIGHT_TYPE)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			strcpy(EdgeWeightType, sub);
			printf("Edge Weight Type is %s \n", EdgeWeightType);
		}
	}

	// Close file
	fclose(fp);

	// Obtain node matrix
	// Calculate amount of rows
	int nodeRows = countMatrixRows(file_name, NODE_COORD_SECTION);
	// Calculate amount of coluns
	int nodeColumns = 3;
	// Get matrix
	nodeMatrix = extractMatrix(file_name, NODE_COORD_SECTION, nodeRows, nodeColumns);
	// Visualize values for node matrix
	printf("Matrix of Nodes has %d rows \n", nodeRows);
	display(nodeMatrix, nodeRows, nodeColumns);

	// Obtain item matrix
	// Calculate amount of rows
	int itemRows = countMatrixRows(file_name, ITEMS_SECTION);
	// Calculate amount of coluns
	int itemColumns = 4;
	// Get matrix
	itemMatrix = extractMatrix(file_name, ITEMS_SECTION, itemRows, itemColumns);
	// Visualize values for item matrix
	printf("Matrix of items has %d rows \n", itemRows);
	display(itemMatrix, itemRows, itemColumns);

	// Calculate Distance Matrix
	float** distanceMatrix;
	distanceMatrix = (float**)malloc(nodeRows * nodeRows * sizeof(float*));
	if (distanceMatrix == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}
	for (int i = 0; i < nodeRows * nodeRows; i++) {
		distanceMatrix[i] = (float*)malloc(nodeColumns * sizeof(float));
		if (distanceMatrix[i] == NULL) {
			fprintf(stderr, "Out of Memory");
			exit(0);
		}
	}

	euclideanDistance(nodeMatrix, nodeMatrix, distanceMatrix, nodeRows, nodeRows * nodeRows);
	display(distanceMatrix, nodeRows * nodeRows, 3);

	// Calculate Distance Matrix in CUDA
	// Define device pointers
	float** devDistanceMatrix;
	int** devNodeMatrix;
	cudaMalloc((void**)&devDistanceMatrix, nodeRows * nodeRows * sizeof(float*));
	cudaMalloc((void**)&devNodeMatrix, sizeof(int*) * nodeRows + sizeof(int) * nodeRows * nodeColumns);
	cudaMemcpy(devNodeMatrix, nodeMatrix, sizeof(int*) * nodeRows + sizeof(int) * nodeRows * nodeColumns, cudaMemcpyHostToDevice);

	// TODO: Implement CUDA Calls

	cudaFree(devNodeMatrix);
	cudaFree(devDistanceMatrix);
	free(distanceMatrix);
	free(nodeMatrix);
	free(itemMatrix);

	/*Initialize CUDA Sub Routines*/
	int count;
	cudaDeviceProp prop;
	cudaGetDeviceCount(&count);
	printf("**********************************************************************************************\n");
	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("GPU: %s\n", prop.name);
		printf("Compute Mode: %d\n", prop.computeMode);
		printf("Max Grid Size: %d\n", prop.maxGridSize);
		printf("Warp Size: %d\n", prop.warpSize);
		printf("Total Global Memory: %zd\n", prop.totalGlobalMem);
		printf("Total Constant Memory: %zd\n", prop.totalConstMem);
		printf("Shared Memory Per Block: %zd\n", prop.sharedMemPerBlock);
		printf("Multiprocessor: %d\n", prop.multiProcessorCount);
		printf("Max Threads Per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
		printf("Max Blocks Per Multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
		printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
		printf("Max Size of Each Dimension of a Block: %d\n", prop.maxThreadsDim);
	}
	printf("**********************************************************************************************\n");

	// End Execution
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

// Helper function for using CUDA to add vectors in parallel.
/*/cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
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
}*/
