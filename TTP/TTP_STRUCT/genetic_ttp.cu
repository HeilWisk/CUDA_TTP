#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <vector>
#include <sstream>

// POPULATION CONTROL
#define MAX_COORD 250
#define POPULATION_SIZE 10 //blockPerGrid*blockPerGrid*BLOCK_SIZE*BLOCK_SIZE
#define BLOCK_SIZE 16
#define NUM_EVOLUTIONS 100
#define MUTATION_RATE 0.05
#define ELITISM true
#define TOURNAMENT_SIZE 128
//BLOCKS
//NUM_THREADS

const int blockPerGrid = 8;

#include "headers/item.h"
#include "headers/node.h"
#include "headers/distance.h"
#include "headers/tour.h"
#include "headers/population.h"
#include "headers/gpu_util.h"
#include "headers/util.h"

#define DIMENSION "DIMENSION:"
#define ITEM_QTY "NUMBER OF ITEMS:"
#define KNAPSACK_CAPACITY "CAPACITY OF KNAPSACK:"
#define MIN_SPEED "MIN SPEED:"
#define MAX_SPEED "MAX SPEED:"
#define RENTING_RATIO "RENTING RATIO:"
#define EDGE_WEIGHT_TYPE "EDGE_WEIGHT_TYPE:"
#define NODE_COORD_SECTION "NODE_COORD_SECTION	(INDEX, X, Y):"
#define ITEMS_SECTION "ITEMS SECTION	(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):"

static void HandleError(cudaError_t err, const char* file, int line) 
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		getchar();
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#pragma region CUDA Kernels

/// <summary>
/// Basic implementation of matrix transpose
/// </summary>
/// <param name="m_dev">- Matrix to be transposed on device memory</param>
/// <param name="t_m_dev">- Matrix Transpose result on device memory</param>
/// <param name="width">- Width of the matrix</param>
/// <param name="height">- Height of the matrix</param>
/// <returns></returns>
__global__ void transpose(node* m_dev, node* t_m_dev, int width, int height) {

	/* Calculate global index for this thread */
	unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

	/* Copy m_dev[rowIdx][colIdx] to t_m_dev[rowIdx][colIdx] */
	if (colIdx < width && rowIdx < height)
	{
		unsigned int index_in = colIdx + width * rowIdx;
		unsigned int index_out = rowIdx + height * colIdx;
		t_m_dev[index_out] = m_dev[index_in];
		if (t_m_dev[index_out].item_qty > 0)
		{
			t_m_dev[index_out].items = m_dev[index_in].items;
		}
		
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
__global__ void matrixTransposeCoalesced(node* m_dev, node* t_m_dev, int width, int height) {

	__shared__ node block[BLOCK_SIZE][BLOCK_SIZE + 1];

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

/// <summary>
/// Kernel to calculate distances between point matrixes
/// </summary>
/// <param name="m_src_dev">- Matrix with source coodinates</param>
/// <param name="m_dst_dev">- Matrix with destination coordinates</param>
/// <param name="m_dist_dev">- Result Matrix with euclidean distances</param>
/// <param name="m_dist_dev_rows">- Result matrix row count</param>
/// <param name="m_dist_dev_cols">- Result matrix column count</param>
/// <returns></returns>
__global__ void matrixDistances(node* m_src_dev, node* m_dst_dev, distance* m_dist_dev, int m_dist_dev_rows, int m_dist_dev_cols) {

	// Define variables
	const unsigned int width = 1;

	// Calculate global indexes
	unsigned int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int colIdx = blockIdx.x * blockDim.x + threadIdx.x;

	// Check boundary conditions
	if (rowIdx < m_dist_dev_rows && colIdx < m_dist_dev_cols)
	{
		// Execute distance calculation
		float value = 0;
		int sourceId = 0;
		int destinyId = 0;
		for (int k = 0; k < width; k++)
		{
			sourceId = m_src_dev[rowIdx * width + k].id;
			destinyId = m_dst_dev[k * m_dist_dev_cols + colIdx].id;
			value += pow(m_dst_dev[k * m_dist_dev_cols + colIdx].x - m_src_dev[rowIdx * width + k].x, 2) + pow(m_dst_dev[k * m_dist_dev_cols + colIdx].y - m_src_dev[rowIdx * width + k].y, 2);
		}
		m_dist_dev[rowIdx * m_dist_dev_cols + colIdx].source = sourceId;
		m_dist_dev[rowIdx * m_dist_dev_cols + colIdx].destiny = destinyId;
		m_dist_dev[rowIdx * m_dist_dev_cols + colIdx].value = sqrt(value);
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="state"></param>
/// <param name="seed"></param>
/// <returns></returns>
__global__ void initCuRand(curandState* state, time_t seed)
{
	// Calculate global index of the threads for the 2D GRID
	// Global index of every block on the grid
	int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;

	if (thread_global_index >= POPULATION_SIZE)
		return;

	curand_init(seed, thread_global_index, 0, &state[thread_global_index]);
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="distance_table"></param>
/// <param name="node_quantity"></param>
/// <returns></returns>
__global__ void evaluatePopulation(population* population, distance* distance_table, const int node_quantity)
{
	// Get thread ID
	// Global index of every block on the grid
	int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;

	if (thread_global_index < POPULATION_SIZE)
		evaluateTour(population->tours[thread_global_index], distance_table, node_quantity);
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="randState"></param>
/// <param name="parents"></param>
/// <param name="node_quantity"></param>
/// <param name="item_quantity"></param>
/// <returns></returns>
__global__ void selection(population* population, curandState* randState, tour* parents, const int node_quantity, const int item_quantity)
{
	// Get thread global id
	// Global index of every block on the grid
	int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;

	if (thread_global_index < POPULATION_SIZE)
	{
		parents[thread_global_index * 2] = tournamentSelection(*population, randState, thread_global_index, node_quantity, item_quantity);
		parents[thread_global_index * 2+1] = tournamentSelection(*population, randState, thread_global_index, node_quantity, item_quantity);
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="parents"></param>
/// <param name="random_state"></param>
/// <param name="distance_table"></param>
/// <param name="index"></param>
/// <returns></returns>
__global__ void crossover(population* population, tour* parents, curandState* random_state, distance* distance_table, int index)
{
	// Get thread global id
	// Global index of every block on the grid
	int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;

	if (thread_global_index < POPULATION_SIZE)
	{
		population->tours[thread_global_index].nodes[0] = parents[2 * thread_global_index].nodes[0];
		node node_1 = getValidNextNode(parents[thread_global_index * 2], population->tours[thread_global_index], population->tours[thread_global_index].nodes[index - 1], index);

	}
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="d_state"></param>
/// <returns></returns>
__global__ void mutate(population* population, curandState* d_state, const int number_nodes)
{
	// Get thread global id
	// Global index of every block on the grid
	int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;

	if (thread_global_index < POPULATION_SIZE)
	{
		// Pick random number between 0 and 1
		curandState local_state = d_state[thread_global_index];

		// If random number is less than mutation rate, perform mutation (swap to nodes in tour)
		if (curand_uniform(&local_state) < MUTATION_RATE)
		{
			int random_number_one = 1 + curand_uniform(&local_state) * (number_nodes - 1.0000001);
			int random_number_two = 1 + curand_uniform(&local_state) * (number_nodes - 1.0000001);

			node temp = population->tours[thread_global_index].nodes[random_number_one];
			population->tours[thread_global_index].nodes[random_number_one] = population->tours[thread_global_index].nodes[random_number_two];
			population->tours[thread_global_index].nodes[random_number_two] = temp;

			d_state[thread_global_index] = local_state;
		}
	}
}

__global__ void nodeTest(node* dev_node, int node_quantity) {
	for (int i = 0; i < node_quantity; i++) 
	{
		printf("dev_node[%d].id: %d ", i, dev_node[i].id);
		printf("x: %lf, ", dev_node[i].x);
		printf("y: %lf\n", dev_node[i].y);
		printf("dev_node[%d].item_qty: %d\n", i, dev_node[i].item_qty);
		if(dev_node[i].item_qty > 0)
		{
			for (int j = 0; dev_node[i].item_qty < node_quantity; j++)
			{
				printf("dev_node[%d].items[%d].id: %d\n", i, j, dev_node[i].items[j].id);
				printf("dev_node[%d].items[%d].node: %d\n", i, j, dev_node[i].items[j].node);
				printf("dev_node[%d].items[%d].value: %d\n", i, j, dev_node[i].items[j].value);
				printf("dev_node[%d].items[%d].weight: %d\n", i, j, dev_node[i].items[j].weight);
				printf("dev_node[%d].items[%d].taken: %d\n", i, j, dev_node[i].items[j].taken);
			}
		}
	}
	printf("\n\n");
}

__global__ void tourTest(tour* tour, int tour_size)
{
	for (int t = 0; t < tour_size; ++t)
	{
		printf(" > tour[%d].fitness: %d\n", t, tour[t].fitness);
		printf(" > tour[%d].total_distance: %d\n", t, tour[t].total_distance);
		printf(" > tour[%d].node_qty: %d\n", t, tour[t].node_qty);
		if (tour[t].node_qty > 0)
		{
			for (int n = 0; n < tour[t].node_qty; ++n)
			{
				printf(" > tour[%d].nodes[%d].id: %d\n", t, n, tour[t].nodes[n].id);
				printf(" > tour[%d].nodes[%d].x: %lf\n", t, n, tour[t].nodes[n].x);
				printf(" > tour[%d].nodes[%d].y: %lf\n", t, n, tour[t].nodes[n].y);
				printf(" > tour[%d].nodes[%d].item_qty: %d\n", t, n, tour[t].nodes[n].item_qty);
				if (tour[t].nodes[n].item_qty > 0)
				{
					for (int i = 0; i < tour[t].nodes[n].item_qty; ++i)
					{
						printf(" > tour[%d].nodes[%d].items[%d].id: %d\n", t, n, i, tour[t].nodes[n].items[i].id);
						printf(" > tour[%d].nodes[%d].items[%d].node: %d\n", t, n, i, tour[t].nodes[n].items[i].node);
						printf(" > tour[%d].nodes[%d].items[%d].taken: %d\n", t, n, i, tour[t].nodes[n].items[i].taken);
						printf(" > tour[%d].nodes[%d].items[%d].value: %f\n", t, n, i, tour[t].nodes[n].items[i].value);
						printf(" > tour[%d].nodes[%d].items[%d].weight: %f\n", t, n, i, tour[t].nodes[n].items[i].weight);
					}
				}
			}
		}
	}
	printf("\n\n");
}

#pragma endregion

#pragma region CUDA Functions

void cudaCheckError()
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		fprintf(stderr, "CUDA Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
	}
}

#pragma endregion

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
/// <param name="rows">- Amount of columns</param>
/// <param name="cols">- Amount of rows</param>
/// <returns>Double pointer matrix of ints</returns>
int** extractMatrixFromFile(const char fileName[], const char sectionName[], int rows, int cols)
{
	FILE* filePtr;
	char str[255], sub[255], * token;
	int lineCount = 0, initialPosition = 0, matrixRow, matrixCol;
	const char openMode[] = "r";

	filePtr = fopen(fileName, openMode);

	// Allocate memory for rows
	int** matrixResult = (int**)malloc(rows * sizeof(int*));
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

/// <summary>
/// Displays a matrix on screen
/// </summary>
/// <param name="matrix">- Matrix to display</param>
/// <param name="rows">- Amount of rows in the matrix</param>
/// <param name="c">- Amount of columns in the matrix</param>
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
/// <param name="c">- Amount of columns in the matrix</param>
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
/// Calculates euclidean distance between a matrix of source points and a matrix of destination points
/// </summary>
/// <param name="srcPoint">- Matrix of source points</param>
/// <param name="dstPoint">- Matrix of destination points</param>
/// <param name="out">- Result matrix with distances</param>
/// <param name="rCount">- Row count</param>
/// <param name="size">- Total size of the result matrix</param>
void euclideanDistanceCPU(node* srcPoint, node* dstPoint, distance* out, int rCount, int size) {
	for (int s = 0; s < size; s++) {
		for (int xSrc = 0; xSrc < rCount; xSrc++) {
			for (int xDst = 0; xDst < rCount; xDst++) {
				out[s].source = srcPoint[xSrc].id;
				out[s].destiny = dstPoint[xDst].id;
				out[s].value = (float)sqrt(pow(dstPoint[xDst].x - srcPoint[xSrc].x, 2) + pow(dstPoint[xDst].y - srcPoint[xSrc].y, 2) * 1.0);
				s++;
			}
		}
	}
}

int main()
{
	/****************************************************************************************************
	* DECLARE VARIABLES
	****************************************************************************************************/

	// File variables
	char file_name[255], str[255], sub[255];
	FILE* fp;
	size_t position;
	const char openMode[] = "r";

	// Problem variables
	int** matrix;
	double knapsack_capacity;
	double minimal_speed;
	double maximun_speed;
	double renting_ratio;
	char edge_weight_type[1000];

	unsigned int node_size;
	unsigned int item_size;
	unsigned int population_size = 1;
	unsigned int tour_size = POPULATION_SIZE;

#pragma region PRINT GPU PROPERTIES

	/****************************************************************************************************
	* PRINT START OF THE PROGRAM
	****************************************************************************************************/
	int count;
	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	printf("****************************************************************************************\n");
	printf("PROPERTIES OF THE GRAPHICAL PROCESSING UNIT\n");
	printf("****************************************************************************************\n");
	for (int i = 0; i < count; i++)
	{
		HANDLE_ERROR(cudaGetDeviceProperties(&properties, i));
		printf("GPU:					%s\n", properties.name);
		printf("Warp Size:				%d\n", properties.warpSize);
		printf("Total Global Memory:			%zd\n", properties.totalGlobalMem);
		printf("Total Constant Memory:			%zd\n", properties.totalConstMem);
		printf("Shared Memory Per Block:		%zd\n", properties.sharedMemPerBlock);
		printf("Multiprocessor:				%d\n", properties.multiProcessorCount);
		printf("Max Threads Per Multiprocessor:		%d\n", properties.maxThreadsPerMultiProcessor);
		printf("Max Blocks Per Multiprocessor:		%d\n", properties.maxBlocksPerMultiProcessor);
		printf("Max Threads Per Block:			%d\n", properties.maxThreadsPerBlock);
	}
	printf("****************************************************************************************\n");
#pragma endregion

#pragma region CAPTURE FILE PATH
	/****************************************************************************************************
	* CAPTURE FILE PATH AND LOAD HIS DATA
	****************************************************************************************************/
	// Ask for the filepath & name where the problem is defined
	printf("Enter name of a file you wish to see\n");
	gets_s(file_name);
	printf("\n");

	// Open the file in read mode
	fp = fopen(file_name, openMode);

	// Validates for errors on file opening
	if (fp == NULL)
	{
		perror("Error while opening the file.\n");
		exit(EXIT_FAILURE);
	}

	// Print headers
	printf("****************************************************************************************\n");
	printf("CONTENTS OF THE FILE:\n");
	printf("****************************************************************************************\n");
	printf("The quantity of lines in the file are:	%d\n", countFileLines(file_name));

	// Obtain general data from file
	while (fgets(str, 100, fp) != NULL) {
		position = findCharacterPosition(str, ':');
		// Extract amount of nodes (cities)
		if (strncmp(str, DIMENSION, strlen(DIMENSION)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			node_size = atoi(sub);
			printf("Nodes (Cities):				%d\n", node_size);
		}
		// Extract the amount of items
		else if (strncmp(str, ITEM_QTY, strlen(ITEM_QTY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			item_size = atoi(sub);
			printf("Item:					%d\n", item_size);
		}
		// Extract the knapsack capacity
		else if (strncmp(str, KNAPSACK_CAPACITY, strlen(KNAPSACK_CAPACITY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			knapsack_capacity = atof(sub);
			printf("Knapsack Capacity:			%lf\n", knapsack_capacity);
		}
		// Extract the minimal speed
		else if (strncmp(str, MIN_SPEED, strlen(MIN_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			minimal_speed = atof(sub);
			printf("Minimum Speed:				%lf\n", minimal_speed);
		}
		// Extract the maximum speed
		else if (strncmp(str, MAX_SPEED, strlen(MAX_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			maximun_speed = atof(sub);
			printf("Maximum Speed:				%lf\n", maximun_speed);
		}
		// Extract the renting ratio
		else if (strncmp(str, RENTING_RATIO, strlen(RENTING_RATIO)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			renting_ratio = atof(sub);
			printf("Renting Ratio:				%lf\n", renting_ratio);
		}
		// Extract the edge weight type
		else if (strncmp(str, EDGE_WEIGHT_TYPE, strlen(EDGE_WEIGHT_TYPE)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			strcpy(edge_weight_type, sub);
			printf("Edge Weight Type is			%s", edge_weight_type);
		}
	}

	// Close file
	fclose(fp);
	printf("****************************************************************************************\n");
	printf("\n");

#pragma endregion

	/****************************************************************************************************
	* PRINT CUDA AND GENETIC VALUES
	****************************************************************************************************/
	printf("****************************************************************************************\n");
	printf("PROPERTIES FOR THE PROBLEM\n");
	printf("****************************************************************************************\n");
	printf("THREADS:				PD\n");
	printf("BLOCKS:					PD\n");
	printf("TOURNAMENT SIZE:			PD\n");
	printf("AMOUNT OF EVOLUTIONS:			PD\n");
	printf("****************************************************************************************\n");

#pragma region POPULATION INITIALIZATION CPU

	/*************************************************************************************************
	* POPULATION INITIALIZATION ON HOST (CPU)
	*************************************************************************************************/
	
	tour initial_tour(node_size, item_size, false);
	population initial_population;

	// Obtain the items
	// Calculate amount of rows
	unsigned int item_rows = countMatrixRows(file_name, ITEMS_SECTION);

	// Validate file consistency
	if (item_rows != item_size)
	{
		perror("The file information is not consistent. Number of items Inconsistency.\n");
		exit(EXIT_FAILURE);
	}

	// Calculate amount of columns
	unsigned int item_columns = 4;

	// Get matrix
	matrix = extractMatrixFromFile(file_name, ITEMS_SECTION, item_size, item_columns);

	// Allocate memory for the array of structs
	item* cpu_item = (item*)malloc(item_size * sizeof(item));
	if (cpu_item == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	// Convert to array of struct
	extractItems(matrix, item_size, cpu_item);

	// Visualize values for item matrix	
	displayItems(cpu_item, item_size);

	// Obtain nodes
	// Calculate amount of nodes
	unsigned int node_rows = countMatrixRows(file_name, NODE_COORD_SECTION);

	// Validate file consistency
	if (node_rows != node_size)
	{
		perror("The file information is not consistent. Number of node Inconsistency.\n");
		exit(EXIT_FAILURE);
	}

	// Calculate amount of columns
	unsigned int node_columns = 3;

	// Get matrix
	matrix = extractMatrixFromFile(file_name, NODE_COORD_SECTION, node_size, node_columns);

	// Allocate memory for the array of structs
	node* cpu_node = (node*)malloc(node_size * sizeof(node));
	if (cpu_node == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}
	// Convert to array of struct
	extractNodes(matrix, node_size, cpu_node);

	// Assign items to node
	assignItems(cpu_item, item_size, cpu_node, node_size);

	// Print node information
	displayNodes(cpu_node, node_size);

	// Assign nodes to tour
	defineInitialTour(initial_tour, node_size, cpu_node);

	// Calculate distance matrix in CPU
	int distance_matrix_size = node_size * node_size;

	// Allocate memory for the distance matrix
	distance* d = (distance*)malloc(distance_matrix_size * sizeof(distance));
	if (d == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	euclideanDistanceCPU(cpu_node, cpu_node, d, node_size, distance_matrix_size);
	displayDistance(d, distance_matrix_size);

	// Initialize population by generating POPULATION_SIZE number of
	// permutations of the initial tour, all starting at the same city
	initializePopulationCPU(initial_population, initial_tour, d, POPULATION_SIZE, node_size);
	//testMemoryAllocationCPU(initial_population, 1);
	printPopulation(initial_population, POPULATION_SIZE);

#pragma endregion

#pragma region POPULATION INITIALIZATION GPU
	/*************************************************************************************************
	* POPULATION INITIALIZATION ON DEVICE (GPU)
	*************************************************************************************************/

	// Setup execution parameters
	//dim3 grid(node_columns / BLOCK_SIZE, node_rows / BLOCK_SIZE, 1);
	dim3 grid(blockPerGrid, blockPerGrid, 1);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

	// Initialize random values
	curandState* d_states;
	HANDLE_ERROR(cudaMalloc((void**)&d_states, sizeof(curandState) * POPULATION_SIZE * node_size));
	initCuRand << <grid, threads >> > (d_states, time(NULL));
	HANDLE_ERROR(cudaDeviceSynchronize());

	/*************************************************************************************************
	* ALLOCATE MEMORY FOR STRUCTS ON DEVICE
	*************************************************************************************************/
	// We are going to start the process of allocation bottom-up, it's say from the inner structure 
	// to the top structure
	// The inner structures are "item" and "node" in which node contains item.

	// Define pointers for device structs
	item* device_item;
	node* device_node;
	tour* device_tour;
	population* device_population;

	// Define pointers for host structs
	item* host_item;
	node* host_node;
	tour* host_tour;
	population* host_population;

	// Allocate host and device memory for population
	HANDLE_ERROR(cudaMalloc((void**)&device_population, sizeof(population) * size_t(population_size)));
	host_population = (population*)malloc(sizeof(population) * size_t(population_size));
	for (int p = 0; p < population_size; ++p)
	{
		host_population[p].tours = (tour*)malloc(sizeof(tour) * size_t(tour_size));
	}

	// Allocate host and device memory for tour
	HANDLE_ERROR(cudaMalloc((void**)&device_tour, sizeof(tour) * size_t(tour_size)));
	host_tour = (tour*)malloc(sizeof(tour) * size_t(tour_size));
	for (int t = 0; t < tour_size; ++t)
	{
		host_tour[t].nodes = (node*)malloc(sizeof(node) * size_t(node_size));
	}

	// Allocate host and device memory for node
	HANDLE_ERROR(cudaMalloc((void**)&device_node, sizeof(node) * size_t(node_size)));
	host_node = (node*)malloc(sizeof(node) * size_t(node_size));
	for (int n = 0; n < node_size; ++n)
	{
		host_node[n].items = (item*)malloc(sizeof(item) * size_t(item_size));
	}

	// Allocate host and device memory for item
	HANDLE_ERROR(cudaMalloc((void**)&device_item, sizeof(item) * size_t(item_size)));
	host_item = (item*)malloc(sizeof(item) * size_t(item_size));

	// Offset pointers
	for (int n = 0; n < node_size; ++n)
	{
		for (int i = 0; i < item_size; ++i)
		{
			if (host_node[n].id == host_item[i].node)
			{
				host_node[n].items = device_item + i;
			}
		}
	}

	for (int t = 0; t < tour_size; ++t)
	{
		host_tour[t].nodes = device_node;
	}

	// Copy host struct with device pointers to device
	//HANDLE_ERROR(cudaMemcpy(device_tour, host_tour, sizeof(tour) * size_t(tour_size), cudaMemcpyHostToDevice));

	for (int p = 0; p < population_size; ++p)
	{
		host_population[p].tours = device_tour;
	}	

	nodeTest << <1, 1 >> > (device_node, node_size);
	HANDLE_ERROR(cudaDeviceSynchronize());

	/*************************************************************************************************
	* GENERATE INITIAL TOUR ON DEVICE
	*************************************************************************************************/
	
	// Define pointers to device
	tour* device_initial_tour;
	node* device_initial_node;
	item* device_initial_item;

	// Define pointers to host
	tour* host_initial_tour;
	node* host_initial_node;
	item* host_initial_item;

	// Allocate memory on device and host for tour
	HANDLE_ERROR(cudaMalloc((void**)&device_initial_tour, sizeof(tour)));
	host_initial_tour = (tour*)malloc(sizeof(tour));
	host_initial_tour->nodes = (node*)malloc(sizeof(node) * size_t(node_size));

	// Copy tour data into host initial tour
	memcpy(host_initial_tour, &initial_tour, sizeof(tour));

	// Allocate memory on device and host for node
	HANDLE_ERROR(cudaMalloc((void**)&device_initial_node, sizeof(node) * size_t(node_size)));
	host_initial_node = (node*)malloc(sizeof(node) * size_t(node_size));
	for (int n = 0; n < node_size; ++n)
	{
		host_initial_node[n].items = (item*)malloc(sizeof(item) * size_t(item_size));
	}

	// Copy node data into host initial node
	memcpy(host_initial_node, cpu_node, sizeof(node) * size_t(node_size));

	// Allocate memory on device and host for item
	HANDLE_ERROR(cudaMalloc((void**)&device_initial_item, sizeof(item) * size_t(item_size)));
	host_initial_item = (item*)malloc(sizeof(item) * size_t(item_size));

	// Copy item data into host initial item
	memcpy(host_initial_item, cpu_item, sizeof(item) * size_t(item_size));

	// Copy data to device pointer
	HANDLE_ERROR(cudaMemcpy(device_initial_item, host_initial_item, sizeof(item) * size_t(item_size), cudaMemcpyHostToDevice));

	// Offset Pointers
	for (int n = 0; n < node_size; ++n)
	{
		for (int i = 0; i < item_size; i++)
		{
			if (host_initial_node[n].id == host_initial_item[i].node)
			{
				host_initial_node[n].items = device_initial_item + i;
			}
		}
	}

	// Copy node data to device pointer
	HANDLE_ERROR(cudaMemcpy(device_initial_node, host_initial_node, sizeof(node) * size_t(node_size), cudaMemcpyHostToDevice));

	for (int t = 0; t < tour_size; t++)
	{
		host_initial_tour[t].nodes = device_initial_node;
	}

	// Copy host initial tour to device initial tour
	HANDLE_ERROR(cudaMemcpy(device_initial_tour, host_initial_tour, sizeof(tour), cudaMemcpyHostToDevice));

	printTour(initial_tour);

	// Test initial tour
	tourTest << <1, 1 >> > (device_initial_tour, 1);
	HANDLE_ERROR(cudaDeviceSynchronize());

	/*END INITIAL TOUR*/

	population* d_initial_population;
	HANDLE_ERROR(cudaMalloc((void**)&d_initial_population, sizeof(population)));

	// 2. Create a separate tour pointer on the host.
	tour* d_tour_ptr;
	HANDLE_ERROR(cudaMalloc((void**)&d_tour_ptr, sizeof(tour) * POPULATION_SIZE));

	// 3. Create a separate node pointer on the host.
	node* d_node_ptr[POPULATION_SIZE];

	// Allocate memory on device according to population size
	for (int i = 0; i < POPULATION_SIZE; ++i)
	{
		// 4. cudaMalloc node storage on the device for node pointer
		HANDLE_ERROR(cudaMalloc((void**)&(d_node_ptr[i]), sizeof(node) * node_size));
		// 5. cudaMemcpy the pointer value of node pointer from host to the device node pointer
		HANDLE_ERROR(cudaMemcpy(&(d_tour_ptr[i].nodes), &(d_node_ptr[i]), sizeof(node*), cudaMemcpyHostToDevice));
		// Optional: Copy an instantiated object on the host to the device pointer
		HANDLE_ERROR(cudaMemcpy(d_node_ptr[i], initial_tour.nodes, sizeof(node) * node_size, cudaMemcpyHostToDevice));
	}
	// 6. cudaMemcpy the pointer value of tour pointer from host to the device population pointer
	HANDLE_ERROR(cudaMemcpy(&(d_initial_population->tours), &d_tour_ptr, sizeof(tour*), cudaMemcpyHostToDevice));

	//testMemoryAllocation << <grid, threads >> > (d_initial_population, 1);

	/********************************************************************************************************************
	* Calculate Distance Matrix in CUDA
	********************************************************************************************************************/
	// First calculate the matrix transpose
	// Define device pointers
	node* d_node_matrix;
	node* d_node_t_matrix;

	// Allocate memory on device
	HANDLE_ERROR(cudaMalloc(&d_node_matrix, node_size * sizeof(node)));
	HANDLE_ERROR(cudaMalloc(&d_node_t_matrix, node_size * sizeof(node)));
	HANDLE_ERROR(cudaMemcpy(d_node_matrix, cpu_node, node_size * sizeof(node), cudaMemcpyHostToDevice));	

	// Execute CUDA Matrix Transposition
	printf("Transponiendo la matrix de nodos de tamaño [%d][%d]\n", node_rows, 1);
	transpose << <grid, threads >> > (d_node_matrix, d_node_t_matrix, node_rows, 1);
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Copy results from device to host
	node* h_node_t_matrix = (node*)malloc(sizeof(node) * node_size);
	HANDLE_ERROR(cudaMemcpy(h_node_t_matrix, d_node_t_matrix, sizeof(node) * node_size, cudaMemcpyDeviceToHost));

	// Show information on screen
	displayNodes(h_node_t_matrix, node_size);

	// Calculate size of distance array
	distance* d_distance;
	int distance_size = node_size * node_size;
	HANDLE_ERROR(cudaMalloc(&d_distance, sizeof(distance) * distance_size));
	printf("Calculando la matriz de distancias en GPU\n");
	matrixDistances << <grid, threads >> > (d_node_matrix, d_node_t_matrix, d_distance, node_size, node_size);
	HANDLE_ERROR(cudaDeviceSynchronize());

	//Copy results from device to host
	distance* h_distance = (distance*)malloc(sizeof(distance) * distance_size);
	HANDLE_ERROR(cudaMemcpy(h_distance, d_distance, sizeof(distance) * distance_size, cudaMemcpyDeviceToHost));

	// Show Data
	displayDistance(h_distance, distance_size);
	
	// Invoke Kernel to generate the initial population on the GPU
	initializePopulationGPU << <grid, threads >> > (d_initial_population, d_distance, node_size, item_rows, d_states);
	HANDLE_ERROR(cudaDeviceSynchronize());

	//Copy results from device to host
	population h_initial_population;
	HANDLE_ERROR(cudaMemcpy(&h_initial_population, d_initial_population, sizeof(population), cudaMemcpyDeviceToHost));
	tour* h_tour_ptr = (tour*)malloc(sizeof(tour) * POPULATION_SIZE);
	HANDLE_ERROR(cudaMemcpy(h_tour_ptr, d_tour_ptr, sizeof(tour) * POPULATION_SIZE, cudaMemcpyDeviceToHost));	
	h_initial_population.tours = h_tour_ptr;
	node* h_node_ptr[POPULATION_SIZE];
	
	for (int p = 0; p < POPULATION_SIZE; ++p)
	{
		h_node_ptr[p] = (node*)malloc(sizeof(node) * node_size);
		HANDLE_ERROR(cudaMemcpy(h_node_ptr[p], d_node_ptr[p], sizeof(node) * node_size, cudaMemcpyDeviceToHost));
		h_initial_population.tours[p].nodes = h_node_ptr[p];
	}

	// Print Result
	printPopulation(h_initial_population, POPULATION_SIZE);
#pragma endregion

#pragma region GPU MEMORY ALLOCATION
	/****************************************************************************************************
	* GPU MEMORY ALLOCATION
	****************************************************************************************************/
	//TODO: Evaluar toda la seccion para determinar que se puede quitar y que no, por ahora solo voy a hacer copy-paste
	population* dev_population;
	HANDLE_ERROR(cudaMalloc((void**)&dev_population, sizeof(population)));

	// Array to store parents selected from tournament selection
	tour* device_parents;
	HANDLE_ERROR(cudaMalloc((void**)&device_parents, sizeof(tour) * POPULATION_SIZE * 2));

	// Cost table for crossover function (SCX Crossover)
	// TODO: Revisar esta memoria dado que la tabla de costos que se tiene elaborada es con base a estructuras y ya esta generada en GPU
	distance* device_cost_table;
	HANDLE_ERROR(cudaMalloc((void**)&device_cost_table, sizeof(distance) * node_size * node_size));

	// Array for random numbers
	curandState* device_state;
	HANDLE_ERROR(cudaMalloc((void**)&device_state, POPULATION_SIZE * sizeof(curandState)));
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Copies data to device for evolution
	HANDLE_ERROR(cudaMemcpy(dev_population, &h_initial_population, sizeof(population), cudaMemcpyHostToDevice));
	// TODO: Revisar con lupa esta linea dado que h_distance esta expresado en otros terminos, especificamente es un arreglo de estructura tipo distancia no flotantes
	HANDLE_ERROR(cudaMemcpy(device_cost_table, &h_distance, sizeof(float) * node_size * node_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaDeviceSynchronize());
#pragma endregion

#pragma region TIMED GPU ALGORITHMS
	/****************************************************************************************************
	* TIMED EXECUTION OF EVOLVE POPULATION ON GPU
	****************************************************************************************************/
	float milliseconds;
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start));

	HANDLE_ERROR(cudaDeviceSynchronize());

	/****************************************************************************************************
	* MAIN LOOP OF TSP
	****************************************************************************************************/
	// Initialize random numbers array for tournament selection	
	initCuRand << <grid, threads >> > (d_states, time(NULL));
	HANDLE_ERROR(cudaDeviceSynchronize());

	// Figure out distance and fitness for each individual in population
	evaluatePopulation << <grid, threads >> > (dev_population, device_cost_table, node_size);
	
	for(int e = 0; e < NUM_EVOLUTIONS; ++e)
	{
		selection << <grid, threads >> > (dev_population, device_state, device_parents, node_size, item_size);

		// Breed the population with tournament selection and SCX crossover
		// Perform computation parallelized, build children iteratively
		for (unsigned int j = 1; j < node_size; ++j)
		{
			crossover << <grid, threads >> > (dev_population, device_parents, device_state, device_cost_table, j);
			
			mutate << <grid, threads >> > (dev_population, device_state, node_size);
			
			evaluatePopulation << <grid, threads >> > (dev_population, device_cost_table, node_size);
		}
	}

	HANDLE_ERROR(cudaEventRecord(stop));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

	// Copy memory back to host
	// TODO: Revisar si es necesaria
	HANDLE_ERROR(cudaMemcpy(&initial_population, dev_population, sizeof(population), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaDeviceSynchronize());
#pragma endregion

#pragma region OUTPUT

	/****************************************************************************************************
	* OUTPUT
	****************************************************************************************************/
	tour fittest = getFittestTour(initial_population.tours, POPULATION_SIZE);	
	printf("%f %f\n", milliseconds / 1000, fittest.total_distance);

#pragma endregion


	/****************************************************************************************************
	* FREE MEMORY
	****************************************************************************************************/
	HANDLE_ERROR(cudaFree(d_node_matrix));
	HANDLE_ERROR(cudaFree(d_node_t_matrix));
	free(h_node_t_matrix);
	HANDLE_ERROR(cudaFree(d_distance));
	free(h_distance);
	free(matrix);
	free(cpu_item);
	free(cpu_node);
	free(d);

	HANDLE_ERROR(cudaDeviceReset());
	// End Execution
	return 0;	
}