#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
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

#define BLOCK_SIZE 16

const int blockPerGrid = 8;

void cudaCheckError()
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		fprintf(stderr, "CUDA Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
	}
}

/// <summary>
/// Function to find a character position in a string
/// </summary>
/// <param name="stringToSearch">- String to search</param>
/// <param name="characterToFind">- Character to find in the string</param>
/// <returns>Position in the string of the character</returns>
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

/// <summary>
/// Extracts a string from another string
/// </summary>
/// <param name="originalString">- Original string</param>
/// <param name="subString">- Resulting Substring</param>
/// <param name="position">- Initial position where the substring is about to begin</param>
/// <param name="length">- Length of the desired substring</param>
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

/// <summary>
/// Function to count the amount of lines in a file
/// </summary>
/// <param name="fileName">- File path and name of the file to evaluate</param>
/// <returns>Amount of lines in the file</returns>
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

/// <summary>
/// Count the rows for a matrix in a file with a given structure
/// </summary>
/// <param name="fileName">- File path and name of the file to evaluate</param>
/// <param name="sectionName">- Section name in the file where the matrix begins</param>
/// <returns>Amount of rows in the matrix</returns>
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

/// <summary>
/// Validates if a file exits
/// </summary>
/// <param name="path">- File path and name of the file</param>
/// <returns>0: File does not exist, 1: File exist</returns>
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

/// <summary>
/// Extracts matrix from a file with a given structure
/// </summary>
/// <param name="fileName">- File path and name</param>
/// <param name="sectionName">- Section name in the file</param>
/// <param name="col">- Amount of columns</param>
/// <returns>- Double pointer matrix of integers</returns>
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

/// <summary>
/// Extracts matrix from a file with a given structure
/// </summary>
/// <param name="fileName">- File path and name</param>
/// <param name="sectionName">- Section name in the file</param>
/// <param name="rows">- Amount of columns</param>
/// <param name="cols">- Amount of rows</param>
/// <returns>- Double pointer matrix of floats</returns>
float** extractMatrix(const char fileName[], const char sectionName[], int rows, int cols)
{
	FILE* filePtr;
	char str[255], sub[255], * token;
	int lineCount = 0, initialPosition = 0, matrixRow, matrixCol;
	const char openMode[] = "r";

	filePtr = fopen(fileName, openMode);

	// Allocate memory for rows
	float **matrixResult = (float**)malloc(rows * sizeof(float*));
	if (matrixResult == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	// Allocate memory for columns
	for (int i = 0; i < rows; i++) {
		matrixResult[i] = (float*)malloc(cols * sizeof(float));
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

/// <summary>
/// Extracts matrix from a file with a given structure
/// </summary>
/// <param name="fileName">- File path and name</param>
/// <param name="sectionName">- Section name in the file</param>
/// <param name="rows">- Amount of columns</param>
/// <param name="cols">- Amount of rows</param>
/// <returns>Matrix pointer</returns>
float* extractMatrixFromFile(const char fileName[], const char sectionName[], int rows, int cols)
{
	FILE* file_ptr;
	char str[255], sub[255], * token;
	int lineCount = 0;
	int initialPosition = 0;
	int matrix_counter = 0;
	int matrix_size = rows * cols;
	const char openMode[] = "r";

	file_ptr = fopen(fileName, openMode);

	// Allocate memory for the result
	float* matrix_result = (float*)malloc(matrix_size * sizeof(float));
	if (matrix_result == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	while (fgets(str, 100, file_ptr) != NULL) {
		if (strncmp(str, sectionName, strlen(sectionName)) == 0) {
			initialPosition = lineCount;
		}
		subString(str, sub, 1, 1);
		if (initialPosition != NULL && lineCount > initialPosition && isdigit(sub[0])) {
			token = strtok(str, "	");
			while (token != NULL)
			{
				matrix_result[matrix_counter] = atoi(token);
				token = strtok(NULL, "	");
				if (matrix_counter < matrix_size)
					matrix_counter++;
			}
		}
		else if (initialPosition != NULL && lineCount > initialPosition && isalpha(sub[0]))
		{
			break;
		}
		lineCount++;
	}

	fclose(file_ptr);

	return matrix_result;
}

/// <summary>
/// Displays a matrix on screen
/// </summary>
/// <param name="matrix">- Matrix to display</param>
/// <param name="rows">- Amount of rows in the matrix</param>
/// <param name="columns">- Amount of columns in the matrix</param>
void display(int** matrix, int rows, int columns) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%d ", matrix[i][j]);			
		}
		printf("\n");
	}
	printf("\n");
}

/// <summary>
/// Displays a matrix on screen
/// </summary>
/// <param name="matrix">- Matrix to display</param>
/// <param name="rows">- Amount of rows in the matrix</param>
/// <param name="columns">- Amount of columns in the matrix</param>
void display(float** matrix, int rows, int columns) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

/// <summary>
/// Displays a matrix on screen
/// </summary>
/// <param name="matrix">- Matrix to display</param>
/// <param name="rows">- Amount of rows in the matrix</param>
/// <param name="columns">- Amount of columns in the matrix</param>
void display(float* matrix, int rows, int columns) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%f	", matrix[i * columns + j]);
		}
		printf("\n");
	}
	printf("\n");
}

/// <summary>
/// Calculates euclidean distance between a matrix of source points and a matrix of destination points
/// </summary>
/// <param name="srcPoint">- Matrix of source points</param>
/// <param name="dstPoint">- Matrix of destination points</param>
/// <param name="out">- Result matrix with distances</param>
/// <param name="rCount">- Row count</param>
/// <param name="size">- Total size of the result matrix</param>
void euclideanDistanceCPU(float** srcPoint, float** dstPoint, float** out, int rCount, int size) {
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

/// <summary>
/// Calculates euclidean distance between a matrix of source points and a matrix of destination points
/// </summary>
/// <param name="srcPoint">- Matrix of source points</param>
/// <param name="dstPoint">- Matrix of destination points</param>
/// <param name="out">- Result matrix with distances</param>
/// <param name="srcSize">- Size of the source matrix</param>
/// <param name="dstSize">- Size of the destination matrix</param>
/// <param name="cols">- Amount of columns</param>
void euclideanDistanceCPU(float* srcPoint, float* dstPoint, float* out, int srcSize, int dstSize, int cols) {
	int s = 0;
	for (int xSrc = 0; xSrc < srcSize; xSrc = xSrc + cols) {
		for (int xDst = 0; xDst < dstSize; xDst = xDst + cols) {
			out[s] = srcPoint[xSrc];
			out[s + 1] = dstPoint[xDst];
			out[s + 2] = sqrt(pow(dstPoint[xDst + 1] - srcPoint[xSrc + 1], 2) + pow(dstPoint[xDst + 2] - srcPoint[xSrc + 2], 2) * 1.0);
			s = s + cols;
		}
	}
}

/// <summary>
/// Function to remove the first column from a matrix
/// </summary>
/// <param name="matrix">Input Matrix</param>
void removeFirstColumn(float* matrix, float* out_matrix, int height, int width) {
	int width_out = width - 1;
	int pos = 0;
	int posM = 0;
	for (int i = 0; i < height; i++)
	{		
		for (int j = 0; j < width_out; j++)
		{
			pos = i * width_out + j;
			posM = i * width + j + 1;
			out_matrix[pos] = matrix[posM];
		}
	}
}

//TODO: Implement a function to remove any column from the matrix

/// <summary>
/// Kernel for matrix multiplication
/// </summary>
/// <param name="m_a_dev">- Input Matrix A</param>
/// <param name="m_b_dev">- Input Matrix B</param>
/// <param name="m_out_dev">- Output Matrix (Result of A X B)</param>
/// <param name="width">- Number of calculations per thread</param>
/// <param name="m_out_dev_rows">- Total of rows of the output matrix (Rows of Matrix A)</param>
/// <param name="m_out_dev_cols">- Total of columns of the output matrix (Columns of Matrix B)</param>
/// <returns></returns>
__global__ void matrixMultiplicationGPU(float* m_a_dev, float* m_b_dev, float* m_out_dev, int width, int m_out_dev_rows, int m_out_dev_cols) {
	
	/* Calculate global indexes*/
	unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

	// Check boundry conditions
	if (rowIdx < m_out_dev_rows && colIdx < m_out_dev_cols)
	{
		// Execute the multiplication for one row and one column
		float value = 0;
		for (int k = 0; k < width; k++)
		{
			value += m_a_dev[rowIdx * m_out_dev_rows + k] * m_b_dev[k * m_out_dev_cols + colIdx];
		}
		m_out_dev[rowIdx * m_out_dev_cols + colIdx] = value;
	}
}

/// <summary>
/// Kernel to calculate distances between point matrixes
/// </summary>
/// <param name="m_src_dev">- Matrix with source coodinates</param>
/// <param name="m_dst_dev">- Matrix with destination coordinates</param>
/// <param name="m_dist_dev">- Result Matrix with euclidean distances</param>
/// <param name="m_dist_dev_rows">- Result matrix row count</param>
/// <param name="m_dist_dev_cols">- Result matrix column count</param>
/// <returns></returns>
__global__ void matrixDistances(float* m_src_dev, float* m_dst_dev, float* m_dist_dev, int m_dist_dev_rows, int m_dist_dev_cols) {

	// Define variables
	const unsigned int width = 2;

	// Calculate global indexes
	unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

	// Check boundary conditions
	if (rowIdx < m_dist_dev_rows && colIdx < m_dist_dev_cols)
	{
		// Execute distance calculation
		float value = 0;
		for (int k = 0; k < width; k++)
		{
			value += pow(m_dst_dev[k * m_dist_dev_cols + colIdx] - m_src_dev[rowIdx * width + k], 2);
		}
		m_dist_dev[rowIdx * m_dist_dev_cols + colIdx] = sqrt(value);
	}
}

/// <summary>
/// Basic implementation of matrix transpose
/// </summary>
/// <param name="m_dev">- Matrix to be transposed on device memory</param>
/// <param name="t_m_dev">- Matrix Transpose result on device memory</param>
/// <param name="width">- Width of the matrix</param>
/// <param name="height">- Height of the matrix</param>
/// <returns></returns>
__global__ void matrixTranspose(float* m_dev, float* t_m_dev, int width, int height) {

	/* Calculate global index for this thread */
	unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

	/* Copy m_dev[rowIdx][colIdx] to t_m_dev[rowIdx][colIdx] */
	if (colIdx < width && rowIdx < height)
	{
		unsigned int index_in = colIdx + width * rowIdx;
		unsigned int index_out = rowIdx + height * colIdx;
		t_m_dev[index_out] = m_dev[index_in];
	}
}

/// <summary>
/// Optimized Kernel to ensure all global reads and writes are coalesced and to avoid bank conflicts in
/// shared memory. This Kernel is up to 11x faster than "matrix_transpose" kernel.
/// </summary>
/// <param name="m_dev">- Matrix to be transposed on device memory</param>
/// <param name="t_m_dev">- Matrix Transpose result on device memory</param>
/// <param name="width">- Width of the matrix</param>
/// <param name="height">- Height of the matrix</param>
/// <returns></returns>
__global__ void matrixTransposeCoalesced(float* m_dev, float* t_m_dev, int width, int height) {

	__shared__ float block[BLOCK_SIZE][BLOCK_SIZE + 1];

	// Read matrix tile into shared memory
	// Load one element per thread from device memory (m_dev) and store it in transposed order in block[][]
	unsigned int colIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	unsigned int rowIdx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	if ((colIdx < width) && (rowIdx < height))
	{
		unsigned int index_in = rowIdx * width + colIdx;
		block[threadIdx.y][threadIdx.x] = m_dev[index_in];
	}

	// Synchronise to ensure allwrites to block[][] have completed
	__syncthreads();

	// Write the transposed matrix tile to global memory (t_m_dev) in linear order
	colIdx = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	rowIdx = blockIdx.x * BLOCK_SIZE + threadIdx.y;
	if ((colIdx < height) && (rowIdx < width))
	{
		unsigned int index_out = rowIdx * height + colIdx;
		t_m_dev[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

int main()
{
	char file_name[255], str[255], sub[255];
	FILE* fp;
	size_t position;
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
	float* node_matrix;
	// Calculate amount of rows
	int node_rows = countMatrixRows(file_name, NODE_COORD_SECTION);
	// Calculate amount of columns
	int node_columns = 3;
	// Calculate node matrix size
	int node_matrix_size = node_columns * node_rows;
	// Get matrix
	node_matrix = extractMatrixFromFile(file_name, NODE_COORD_SECTION, node_rows, node_columns);
	// Visualize values for node matrix
	printf("Matrix of Nodes has %d rows \n\n", node_rows);
	printf("INDEX	X	Y\n");
	display(node_matrix, node_rows, node_columns);

	// Obtain item matrix
	float* item_matrix;
	// Calculate amount of rows
	int item_rows = countMatrixRows(file_name, ITEMS_SECTION);
	// Calculate amount of coluns
	int item_columns = 4;
	// Get matrix
	item_matrix = extractMatrixFromFile(file_name, ITEMS_SECTION, item_rows, item_columns);
	// Visualize values for item matrix
	printf("Matrix of items has %d rows \n\n", item_rows);
	printf("INDEX	PROFIT	WEIGHT	ASSIGNED NODE\n");
	display(item_matrix, item_rows, item_columns);

	// Calculate Distance Matrix in CPU
	float* distance_matrix;
	int distance_matrix_size = node_rows * node_rows * node_columns;
	distance_matrix = (float*)malloc(distance_matrix_size * sizeof(float));
	if (distance_matrix == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	euclideanDistanceCPU(node_matrix, node_matrix, distance_matrix, node_matrix_size, node_matrix_size, node_columns);
	printf("SOURCE	DESTINY	DISTANCE\n");
	display(distance_matrix, node_rows * node_rows, 3);	

	// Remove column
	// Define new matrix
	float* coordinate_matrix;
	int coordinate_matrix_size = node_rows * (node_columns - 1);
	coordinate_matrix = (float*)malloc(coordinate_matrix_size * sizeof(float));
	if (coordinate_matrix == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}
	removeFirstColumn(node_matrix, coordinate_matrix, node_rows, node_columns);
	printf("X	Y\n");
	display(coordinate_matrix, node_rows, node_columns - 1);

	// Calculate Distance Matrix in CUDA
	// Define device pointers
	float* d_distance_matrix;
	int distance_matrix_size_gpu = node_rows * node_rows * (node_columns-1);
	float* d_coordinate_matrix;
	float* d_coordinate_t_matrix;
	
	cudaMalloc(&d_coordinate_matrix, sizeof(float) * coordinate_matrix_size);
	cudaMalloc(&d_coordinate_t_matrix, sizeof(float) * coordinate_matrix_size);
	cudaMemcpy(d_coordinate_matrix, coordinate_matrix, sizeof(float) * coordinate_matrix_size, cudaMemcpyHostToDevice);

	// Setup execution parameters
	//dim3 grid(node_columns / BLOCK_SIZE, node_rows / BLOCK_SIZE, 1);
	dim3 grid(blockPerGrid, blockPerGrid, 1);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

	printf("Transponiendo la matrix de coordenadas de tamaño [%d][%d]\n", node_rows, node_columns-1);
	matrixTranspose << <grid, threads >> > (d_coordinate_matrix, d_coordinate_t_matrix, node_columns-1, node_rows);
	cudaThreadSynchronize();

	//Copy results from device to host
	float* h_coordinate_t_matrix = (float*)malloc(sizeof(float) * coordinate_matrix_size);
	cudaMemcpy(h_coordinate_t_matrix, d_coordinate_t_matrix, sizeof(float) * coordinate_matrix_size, cudaMemcpyDeviceToHost);

	display(h_coordinate_t_matrix, node_columns-1, node_rows);

	cudaMalloc(&d_distance_matrix, sizeof(float)* distance_matrix_size_gpu);
	printf("Calculando la matriz de distancias\n");
	matrixDistances << <grid, threads >> > (d_coordinate_matrix, d_coordinate_t_matrix, d_distance_matrix, node_rows, node_rows);
	cudaThreadSynchronize();

	//Copy results from device to host
	float* h_distance_matrix = (float*)malloc(sizeof(float) * distance_matrix_size_gpu);
	cudaMemcpy(h_distance_matrix, d_distance_matrix, sizeof(float)* distance_matrix_size_gpu, cudaMemcpyDeviceToHost);

	display(h_distance_matrix, node_rows, node_rows);

	cudaFree(d_distance_matrix);
	cudaFree(d_coordinate_matrix);
	cudaFree(d_coordinate_t_matrix);
	free(coordinate_matrix);
	free(h_coordinate_t_matrix);
	free(distance_matrix);
	free(node_matrix);
	free(item_matrix);
	
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
}