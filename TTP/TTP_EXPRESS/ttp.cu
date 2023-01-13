﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <vector>

#include "headers/helper_functions.h"
#include "headers/helper_cuda.h"
#include "headers/config.h"
#include "headers/item.cuh"
#include "headers/node.cuh"
#include "headers/params.h"
#include "headers/distance.cuh"
#include "headers/greedy.h"
#include "headers/tour.cuh"
#include "headers/population.cuh"
#include "headers/genetic.cuh"
#include "headers/File.h"

#define NAME "PROBLEM NAME:"
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

	if (thread_global_index >= TOURS)
		return;

	curand_init(seed, thread_global_index, 0, &state[thread_global_index]);
}

/// <summary>
/// 
/// </summary>
/// <param name="tour"></param>
/// <param name="tour_size"></param>
/// <returns></returns>
__global__ void tourTest(tour* tour, int tour_size)
{
	for (int t = 0; t < tour_size; ++t)
	{
		printf(" > tour[%d].fitness: %f\n", t, tour[t].fitness);
		printf(" > tour[%d].total_distance: %f\n", t, tour[t].total_distance);
		for (int n = 0; n < CITIES; ++n)
		{
			printf(" > tour[%d].nodes[%d].id: %d\n", t, n, tour[t].nodes[n].id);
			printf(" > tour[%d].nodes[%d].x: %lf\n", t, n, tour[t].nodes[n].x);
			printf(" > tour[%d].nodes[%d].y: %lf\n", t, n, tour[t].nodes[n].y);
			for (int i = 0; i < ITEMS; ++i)
			{
				printf(" > tour[%d].nodes[%d].items[%d].id: %d\n", t, n, i, tour[t].nodes[n].items[i].id);
				printf(" > tour[%d].nodes[%d].items[%d].node: %d\n", t, n, i, tour[t].nodes[n].items[i].node);
				printf(" > tour[%d].nodes[%d].items[%d].pw_ratio: %f\n", t, n, i, tour[t].nodes[n].items[i].pw_ratio);
				printf(" > tour[%d].nodes[%d].items[%d].value: %f\n", t, n, i, tour[t].nodes[n].items[i].value);
				printf(" > tour[%d].nodes[%d].items[%d].weight: %f\n", t, n, i, tour[t].nodes[n].items[i].weight);
			}
		}
	}
	printf("\n\n");
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <returns></returns>
__global__ void populationTest(population* population)
{
	for (int p = 0; p < POPULATION_SIZE; ++p)
	{
		for (int t = 0; t < TOURS; ++t)
		{
			printf(" > population[%d].tours[%d].fitness: %f\n", p, t, population[p].tours[t].fitness);
			printf(" > population[%d].tours[%d].total_distance: %f\n", p, t, population[p].tours[t].total_distance);
			for (int n = 0; n < CITIES; ++n)
			{
				if (population[p].tours[t].nodes[n].id > 0)
				{
					printf(" > population[%d].tours[%d].nodes[%d].id: %d\n", p, t, n, population[p].tours[t].nodes[n].id);
					printf(" > population[%d].tours[%d].nodes[%d].x: %lf\n", p, t, n, population[p].tours[t].nodes[n].x);
					printf(" > population[%d].tours[%d].nodes[%d].y: %lf\n", p, t, n, population[p].tours[t].nodes[n].y);
					for (int i = 0; i < ITEMS; ++i)
					{
						if (population[p].tours[t].nodes[n].items[i].id >= 0)
						{
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].id: %d\n", p, t, n, i, population[p].tours[t].nodes[n].items[i].id);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].node: %d\n", p, t, n, i, population[p].tours[t].nodes[n].items[i].node);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].pw_ratio: %f\n", p, t, n, i, population[p].tours[t].nodes[n].items[i].pw_ratio);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].value: %f\n", p, t, n, i, population[p].tours[t].nodes[n].items[i].value);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].weight: %f\n", p, t, n, i, population[p].tours[t].nodes[n].items[i].weight);
						}
					}
				}
			}
		}
	}
	printf("\n\n");
}

/// <summary>
/// 
/// </summary>
/// <param name="initial_population"></param>
/// <param name="distances"></param>
/// <param name="node_quantity"></param>
/// <param name="item_quantity"></param>
/// <param name="state"></param>
/// <returns></returns>
__global__ void initPopulationGPU(population* initial_population, tour* initial_tour, const int population_size, curandState* state)
{
	node temp;

	// Get thread ID
	// Global index of every block on the grid
	unsigned int block_number_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
	// Global index of every thread in block
	unsigned int thread_number_in_block = threadIdx.x + blockDim.x * threadIdx.y;
	// Number of thread per block
	unsigned int threads_per_block = blockDim.x * blockDim.y;
	// Global index of every thread on the grid
	unsigned int thread_global_index = block_number_in_grid * threads_per_block + thread_number_in_block;

	curandState local_state = state[thread_global_index];

	// Set the tours
	for (int p = 0; p < population_size; ++p)
	{
		for (int n = 0; n < CITIES; ++n)
		{
			initial_population->tours[thread_global_index].nodes[n] = initial_tour[p].nodes[n];
		}
	}

	for (int j = 1; j < CITIES; ++j)
	{
		int random_position = 1 + (curand(&local_state) % (CITIES - 1));

		temp = initial_population->tours[thread_global_index].nodes[j];
		for (int k = 0; k < ITEMS; ++k)
		{
			temp.items[k] = initial_population->tours[thread_global_index].nodes[j].items[k];
		}

		printf(" > thread_global_index: %d > %d cambia con %d\n", thread_global_index, j, random_position);
		printf(" > thread_global_index: %d > El id de %d es: %d\n", thread_global_index, j, initial_population->tours[thread_global_index].nodes[j].id);
		printf(" > thread_global_index: %d > El id de %d es: %d\n", thread_global_index, random_position, initial_population->tours[thread_global_index].nodes[random_position].id);

		initial_population->tours[thread_global_index].nodes[j] = initial_population->tours[thread_global_index].nodes[random_position];
		for (int l = 0; l < ITEMS; ++l)
		{
			initial_population->tours[thread_global_index].nodes[j].items[l] = initial_population->tours[thread_global_index].nodes[random_position].items[l];
		}

		initial_population->tours[thread_global_index].nodes[random_position] = temp;
		for (int m = 0; m < ITEMS; ++m)
		{
			initial_population->tours[thread_global_index].nodes[random_position].items[m] = temp.items[m];
		}

		printf(" > thread_global_index: %d > initial_population->tours[%d].nodes[%d]: %d\n", thread_global_index, thread_global_index, j, initial_population->tours[thread_global_index].nodes[j].id);
		printf(" > thread_global_index: %d > initial_population->tours[%d].nodes[%d]: %d\n", thread_global_index, thread_global_index, random_position, initial_population->tours[thread_global_index].nodes[random_position].id);
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
		for (int i = 0; i > ITEMS; ++i)
		{
			t_m_dev[index_out].items[i] = m_dev[index_in].items[i];
		}
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
/// <param name="population"></param>
/// <param name="distanceTable"></param>
/// <returns></returns>
__global__ void evaluatePopulation(population* population, distance* distanceTable)
{
	// Get Thread (particle) ID
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= TOURS)
		return;

	evaluateTour(population->tours[tid], distanceTable);
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="distanceTable"></param>
/// <returns></returns>
__global__ void evaluatePopulation(population* population, parameters problem_parameters)
{
	// Get Thread (particle) ID
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= TOURS)
		return;

	evaluateTour(population->tours[tid], problem_parameters);
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="randomState"></param>
/// <param name="parents"></param>
/// <returns></returns>
__global__ void selection(population* population, curandState* randomState, tour* parents)
{
	// Get thread (particle) ID
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= TOURS)
		return;

	parents[tid * 2] = tournamentSelection(*population, randomState, tid);
	parents[tid * 2 + 1] = tournamentSelection(*population, randomState, tid);
}

/// <summary>
/// 
/// </summary>
/// <param name="population"></param>
/// <param name="parents"></param>
/// <param name="randomState"></param>
/// <param name="distanceTable"></param>
/// <param name="index"></param>
/// <returns></returns>
__global__ void crossover(population* population, tour* parents, curandState* randomState, distance* distanceTable, int index)
{
	// Get thread (particle) ID
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= TOURS)
		return;

	population->tours[tid].nodes[0] = parents[2 * tid].nodes[0];

	node nodeOne = getValidNextNode(parents[tid * 2], population->tours[tid], population->tours[tid].nodes[index - 1], index);
	node nodeTwo = getValidNextNode(parents[tid * 2 + 1], population->tours[tid], population->tours[tid].nodes[index - 1], index);

	// Compare the two nodes from parents to the last node that was chosen in the child
	if (distanceTable[nodeOne.id * CITIES + population->tours[tid].nodes[index - 1].id].value <= distanceTable[nodeTwo.id * CITIES + population->tours[tid].nodes[index - 1].id].value)
		population->tours[tid].nodes[index] = nodeOne;
	else
		population->tours[tid].nodes[index] = nodeTwo;
}

/// <summary>
/// Kernel to perform mutation (Swap to random nodes)
/// </summary>
/// <param name="population"></param>
/// <param name="randomState"></param>
/// <returns></returns>
__global__ void mutation(population* population, curandState* randomState)
{
	// Get thread (particle) ID
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= TOURS)
		return;

	// Pick a radom number between zero (0) and one (1)
	curandState localState = randomState[tid];

	// Only mutate by random chance: If random number is less than the mutation rate the mutation is executed (swap to nodes)
	if (curand_uniform(&localState) < MUTATION_RATE)
	{
		// Don´t mutate the first node in the route
		int random_number_one = 1 + curand_uniform(&localState) * (CITIES - 1.000001);
		int random_number_two = 1 + curand_uniform(&localState) * (CITIES - 1.000001);

		node temp_node = population->tours[tid].nodes[random_number_one];
		population->tours[tid].nodes[random_number_one] = population->tours[tid].nodes[random_number_two];
		population->tours[tid].nodes[random_number_two] = temp_node;

		randomState[tid] = localState;
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

	// Problem params
	parameters problem;
	int** matrix;
	char edge_weight_type[1000];

	unsigned int population_size = POPULATION_SIZE;
	unsigned int tour_size = TOURS;

#pragma region PRINT GPU PROPERTIES

	/****************************************************************************************************
	* PRINT START OF THE PROGRAM
	****************************************************************************************************/
	int deviceCount = 0;
	cudaDeviceProp properties;
	cudaError_t deviceErr = cudaGetDeviceCount(&deviceCount);
	if (deviceCount > 0 && deviceErr == cudaSuccess && !NO_GPU)
	{
		printf("****************************************************************************************\n");
		printf("PROPERTIES OF THE GRAPHICAL PROCESSING UNIT\n");
		printf("****************************************************************************************\n");
		for (int i = 0; i < deviceCount; i++)
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
	}
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
		// Extract problem name
		if (strncmp(str, NAME, strlen(NAME)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position - 1);
			strcpy(problem.name, sub);
			createFile(problem.name);
		}
		// Extract amount of nodes (cities)
		if (strncmp(str, DIMENSION, strlen(DIMENSION)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.cities_amount = atoi(sub);
			printf("Nodes (Cities):				%d\n", problem.cities_amount);
		}
		// Extract the amount of items
		else if (strncmp(str, ITEM_QTY, strlen(ITEM_QTY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.items_amount = atoi(sub);
			printf("Item:					%d\n", problem.items_amount);
		}
		// Extract the knapsack capacity
		else if (strncmp(str, KNAPSACK_CAPACITY, strlen(KNAPSACK_CAPACITY)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.knapsack_capacity = atof(sub);
			printf("Knapsack Capacity:			%lf\n", problem.knapsack_capacity);
		}
		// Extract the minimal speed
		else if (strncmp(str, MIN_SPEED, strlen(MIN_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.min_speed = atof(sub);
			printf("Minimum Speed:				%lf\n", problem.min_speed);
		}
		// Extract the maximum speed
		else if (strncmp(str, MAX_SPEED, strlen(MAX_SPEED)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.max_speed = atof(sub);
			printf("Maximum Speed:				%lf\n", problem.max_speed);
		}
		// Extract the renting ratio
		else if (strncmp(str, RENTING_RATIO, strlen(RENTING_RATIO)) == 0)
		{
			subString(str, sub, position + 1, strlen(str) - position);
			problem.renting_ratio = atof(sub);
			printf("Renting Ratio:				%lf\n", problem.renting_ratio);
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
	printf("THREADS:				%d\n", THREADS);
	printf("BLOCKS:					%d\n", BLOCKS);
	printf("TOURNAMENT SIZE:			%d\n", TOURNAMENT_SIZE);
	printf("AMOUNT OF EVOLUTIONS:			%d\n", NUM_EVOLUTIONS);
	printf("****************************************************************************************\n");

#pragma region POPULATION INITIALIZATION CPU

	/*************************************************************************************************
	* POPULATION INITIALIZATION ON HOST (CPU)
	*************************************************************************************************/

	tour initial_tour;
	population initial_population_cpu;
	population initial_population_gpu;

	// Obtain the items
	// Calculate amount of rows
	unsigned int item_rows = countMatrixRows(file_name, ITEMS_SECTION);

	// Validate file consistency
	if (item_rows != problem.items_amount)
	{
		perror("The file information is not consistent. Number of items Inconsistency.\n");
		exit(EXIT_FAILURE);
	}

	// Calculate amount of columns
	unsigned int item_columns = 4;

	// Get matrix
	matrix = extractMatrixFromFile(file_name, ITEMS_SECTION, problem.items_amount, item_columns);

	// Allocate memory for the array of structs
	item* cpu_item = (item*)malloc(problem.items_amount * sizeof(item));
	if (cpu_item == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	// Convert to array of struct
	extractItems(matrix, problem.items_amount, cpu_item);

	// Visualize values for item matrix	
	displayItems(cpu_item, problem.items_amount);

	// Obtain nodes
	// Calculate amount of nodes
	unsigned int node_rows = countMatrixRows(file_name, NODE_COORD_SECTION);

	// Validate file consistency
	if (node_rows != problem.cities_amount)
	{
		perror("The file information is not consistent. Number of node Inconsistency.\n");
		exit(EXIT_FAILURE);
	}

	// Calculate amount of columns
	unsigned int node_columns = 3;

	// Get matrix
	matrix = extractMatrixFromFile(file_name, NODE_COORD_SECTION, problem.cities_amount, node_columns);

	// Allocate memory for the array of structs
	node* cpu_node = (node*)malloc(problem.cities_amount * sizeof(node));
	if (cpu_node == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}
	// Convert to array of struct
	extractNodes(matrix, problem.cities_amount, cpu_node);

	// Assign items to node
	assignItems(cpu_item, cpu_node);

	// Print node information
	displayNodes(cpu_node, problem.cities_amount);

	// Assign nodes to tour
	defineInitialTour(initial_tour, problem, cpu_node);

	// Calculate distance matrix in CPU
	int distance_matrix_size = problem.cities_amount * problem.cities_amount;

	// Allocate memory for the distance matrix
	distance* d = (distance*)malloc(distance_matrix_size * sizeof(distance));
	if (d == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	euclideanDistanceCPU(cpu_node, cpu_node, d, problem.cities_amount, distance_matrix_size);
	displayDistance(d, distance_matrix_size);

	// Initialize population by generating POPULATION_SIZE number of permutations of the initial tour, all starting at the same city
	initializePopulation(initial_population_cpu, initial_tour, problem);

	saveInitialPopulation(problem.name, initial_population_cpu, problem, NO_CUDA);

	printPopulation(initial_population_cpu);

#pragma endregion

#pragma region POPULATION INITIALIZATION GPU

	/*************************************************************************************************
	* POPULATION INITIALIZATION ON DEVICE (GPU)
	*************************************************************************************************/

	// Device Variables
	population* device_population;
	tour* device_parents;
	node* device_node_matrix;
	node* device_node_t_matrix;
	distance* device_distance;
	curandState* device_states;

	float milliseconds = 0;
	cudaEvent_t start, stop;

	if (deviceCount > 0 && deviceErr == cudaSuccess && !NO_GPU)
	{
		// Create output file
		char file_name[] = "CUDA_";
		strcat(file_name, problem.name);
		createFile(file_name);

		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		checkCudaErrors(cudaEventRecord(start));

		/*************************************************************************************************
		* ALLOCATE MEMORY FOR STRUCTS ON DEVICE
		*************************************************************************************************/

		// Allocate device memory for population
		HANDLE_ERROR(cudaMalloc((void**)&device_population, sizeof(population) * size_t(population_size)));

		// Allocate device memory for parents selected from tournament selection
		HANDLE_ERROR(cudaMalloc((void**)&device_parents, sizeof(tour) * size_t(tour_size) * SELECTED_PARENTS));

		// Allocate device memory for node matrix, node matrix transpose and distance matrix
		HANDLE_ERROR(cudaMalloc(&device_node_matrix, size_t(problem.cities_amount) * sizeof(node)));
		HANDLE_ERROR(cudaMalloc(&device_node_t_matrix, size_t(problem.cities_amount) * sizeof(node)));
		HANDLE_ERROR(cudaMalloc(&device_distance, sizeof(distance) * CITIES * CITIES));

		// Allocate device memory for states
		HANDLE_ERROR(cudaMalloc((void**)&device_states, sizeof(curandState) * TOURS * size_t(problem.cities_amount)));

		/*************************************************************************************************
		* COPY HOST MEMORY TO DEVICE
		*************************************************************************************************/

		// Copy node data
		HANDLE_ERROR(cudaMemcpy(device_node_matrix, cpu_node, size_t(problem.cities_amount) * sizeof(node), cudaMemcpyHostToDevice));

		/*************************************************************************************************
		* INITIALIZE RANDOM VALUES
		*************************************************************************************************/
		initCuRand << <BLOCKS, THREADS >> > (device_states, time(NULL));
		cudaDeviceSynchronize();

		checkCudaErrors(cudaEventRecord(stop));
		checkCudaErrors(cudaEventSynchronize(stop));
		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
		// Compute and print the performance
		printf("Time= %.3f msec\n", msecTotal);

		printTour(initial_tour, CITIES);

		initializePopulationCuda << <BLOCKS, THREADS >> > (device_population, initial_tour, problem, device_states);
		cudaDeviceSynchronize();

		/*************************************************************************************************
		* TODO: REMOVE THIS SECTION IF EVERITHING GOES WELL
		*************************************************************************************************/
		//populationTest << <1, 1 >> > (device_population);
		
		/*************************************************************************************************
		* COPY RESULTS TO HOST
		*************************************************************************************************/
		cudaMemcpy(&initial_population_gpu, device_population, sizeof(population), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		/*************************************************************************************************
		* OUTPUT
		*************************************************************************************************/
		printPopulation(initial_population_gpu);
		saveInitialPopulation(problem.name, initial_population_gpu, problem, CUDA);
	}
#pragma endregion

#pragma region DISTANCE MATRIX GPU

	if (deviceCount > 0 && deviceErr == cudaSuccess && !NO_GPU)
	{
		/*************************************************************************************************
		* CALCULATE DISTANCE MATRIX IN CUDA
		*************************************************************************************************/

		// Execute CUDA Matrix Transposition
		printf("Transponiendo la matrix de nodos de tamaño [%d][%d]\n", node_rows, 1);
		transpose << <BLOCKS, THREADS >> > (device_node_matrix, device_node_t_matrix, node_rows, 1);
		cudaDeviceSynchronize();

		// Copy results from device to host
		node* h_node_t_matrix = (node*)malloc(sizeof(node) * problem.cities_amount);
		HANDLE_ERROR(cudaMemcpy(h_node_t_matrix, device_node_t_matrix, sizeof(node) * problem.cities_amount, cudaMemcpyDeviceToHost));

		// Show information on screen
		displayNodes(h_node_t_matrix, problem.cities_amount);

		// TODO: FIX GRID AND THREADS AND MATRIXDISTANCES KERNEL
		dim3 grid(8, 8, 1);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

		printf("Calculando la matriz de distancias en GPU\n");
		matrixDistances << <grid, threads >> > (device_node_matrix, device_node_t_matrix, device_distance, problem.cities_amount, problem.cities_amount);
		cudaDeviceSynchronize();

		//Copy results from device to host
		distance* h_distance = (distance*)malloc(sizeof(distance) * CITIES * CITIES);
		HANDLE_ERROR(cudaMemcpy(h_distance, device_distance, sizeof(distance) * CITIES * CITIES, cudaMemcpyDeviceToHost));

		// Show Data
		displayDistance(h_distance, CITIES * CITIES);
	}
#pragma endregion

	/*************************************************************************************************
	* EVOLVE POPULATION
	*************************************************************************************************/

	if (deviceCount > 0 && deviceErr == cudaSuccess && !NO_GPU)
	{
		// Figure out fitness and distance for each individual in population
		evaluatePopulation << <BLOCKS, THREADS >> > (device_population, problem);
		cudaDeviceSynchronize();
	}

	cudaError_t err = cudaSuccess;

	tour* host_parents = (tour*)malloc(SELECTED_PARENTS * sizeof(tour));
	if (host_parents == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}

	for (int i = 0; i < NUM_EVOLUTIONS; ++i)
	{
		if (deviceCount > 0 && deviceErr == cudaSuccess && !NO_GPU)
		{
			selection << <BLOCKS, THREADS >> > (device_population, device_states, device_parents);
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "Selection Kernel: %s\n", cudaGetErrorString(err));
				exit(0);
			}

			// Breed the population with tournament selection and SCX crossover perform computation parallelized, build children iteratively
			for (int j = 1; j < CITIES; ++j)
			{
				crossover << <BLOCKS, THREADS >> > (device_population, device_parents, device_states, device_distance, j);
				//HANDLE_ERROR(cudaDeviceSynchronize());
				err = cudaGetLastError();
				if (err != cudaSuccess) {
					fprintf(stderr, "Crossover Kernel: %s\n", cudaGetErrorString(err));
					exit(0);
				}
			}

			// Perform mutation
			mutation << <BLOCKS, THREADS >> > (device_population, device_states);
			//HANDLE_ERROR(cudaDeviceSynchronize());
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "Mutation Kernel: %s\n", cudaGetErrorString(err));
				exit(0);
			}

			// Evaluate the resulting population
			evaluatePopulation << <BLOCKS, THREADS >> > (device_population, device_distance);
			//HANDLE_ERROR(cudaDeviceSynchronize());
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				fprintf(stderr, "Evaluate Population Kernel: %s\n", cudaGetErrorString(err));
				exit(0);
			}
		}
		
		// CPU GA
		selection(initial_population_cpu, host_parents);

		// Breed the population with tournament selection and SCX crossover perform computation parallelized, build children iteratively
		int* child = (int*)malloc(CITIES + 1 * sizeof(int));
		orderedCrossover (child, host_parents);
	}

	if (deviceCount > 0 && deviceErr == cudaSuccess && !NO_GPU)
	{
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);

		/*************************************************************************************************
		* COPY RESULTS TO HOST
		*************************************************************************************************/
		cudaMemcpy(&initial_population_gpu, device_population, sizeof(population), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}

	/*************************************************************************************************
	* OUTPUT
	*************************************************************************************************/
	printPopulation(initial_population_cpu);
	tour fittestOnEarth = getFittestTour(initial_population_cpu.tours, TOURS);
	printf("TIME: %f\n", milliseconds);
	printf("FITNESS: %f\n", fittestOnEarth.fitness);
	printf("PROFIT: %f\n", fittestOnEarth.profit);
	printf("ROUTE: %d", fittestOnEarth.nodes[0].id);
	for (int i = 1; i < CITIES + 1; ++i)
		printf(" > %d", fittestOnEarth.nodes[i].id);
	printf("\n");
	printf("ITEMS: %d", fittestOnEarth.item_picks[0].id);
	for (int i = 1; i < ITEMS; ++i)
	{
		if (fittestOnEarth.item_picks[i].id > 0)
			printf(" > %d", fittestOnEarth.item_picks[i].id);
	}
	printf("\n");

	if (deviceCount > 0 && deviceErr == cudaSuccess && !NO_GPU)
	{
		/*************************************************************************************************
		* RELEASE CUDA MEMORY
		*************************************************************************************************/
		cudaFree(device_population);
		cudaFree(device_parents);
		cudaFree(device_node_matrix);
		cudaFree(device_node_t_matrix);
		cudaFree(device_distance);
		cudaFree(device_states);
	}
	return 0;
}