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

//cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

/*__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}*/

#pragma region Struct Definition

// Define struct for City information
struct node{
	int id;
	float x;
	float y;
};

// Define Struct for item information
struct item{
	int id;
	float w;
	float v;
	int lId;
};

// Define struct for distances
struct distance{
	int srcId;
	int dstId;
	float d;
};

#pragma endregion

#pragma region CUDA Kernels

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

int** extractMatrix(const char fileName[], const char sectionName[], int rows, int cols)
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

void extractNodes(int** matrix, int rows, node* c) {
	for (int i = 0; i < rows; i++) {
		c[i].id = matrix[i][0];
		c[i].x = (float)matrix[i][1];
		c[i].y = (float)matrix[i][2];
	}
}

void extractItems(int** matrix, int rows, item* i) {
	for (int s = 0; s < rows; s++) {
		i[s].id = matrix[s][0];
		i[s].v = (float)matrix[s][1];
		i[s].w = (float)matrix[s][2];
		i[s].lId = matrix[s][3];
	}
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

void display(node* c, int size) {
	printf("ID	X	Y\n");
	for (int i = 0; i < size; i++) {
		printf("%d	%f	%f\n", c[i].id, c[i].x, c[i].y);
	}
	printf("\n");
}

void display(item* c, int size) {
	printf("ID	X	Y	LOC\n");
	for (int i = 0; i < size; i++) {
		printf("%d	%f	%f	%d\n", c[i].id, c[i].v, c[i].w, c[i].lId);
	}
	printf("\n");
}

void display(distance* d, int size) {
	printf("srcId	dstId	d\n");
	for (int i = 0; i < size; i++) {
		printf("%d	%d	%f\n", d[i].srcId, d[i].dstId, d[i].d);
	}
	printf("\n");
}

void euclideanDistance(node* srcPoint, node* dstPoint, distance* out, int rCount, int size) {
	for (int s = 0; s < size; s++) {
		for (int xSrc = 0; xSrc < rCount; xSrc++) {
			for (int xDst = 0; xDst < rCount; xDst++) {
				out[s].srcId = srcPoint[xSrc].id;
				out[s].dstId = dstPoint[xDst].id;
				out[s].d = (float)sqrt(pow(dstPoint[xDst].x - srcPoint[xSrc].x, 2) + pow(dstPoint[xDst].y - srcPoint[xSrc].y, 2) * 1.0);
				s++;
			}
		}
	}
}

__global__ void euclideanDistanceStruct(int** srcPts, int** dstPts, float** outDistMat, int rQty, int size) {
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
	int** matrix;
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

	// Close file and free memory
	fclose(fp);

	// Obtain nodes
	// Calculate amount of rows
	int nodeRows = countMatrixRows(file_name, NODE_COORD_SECTION);
	// Calculate amount of columns
	int nodeColumns = 3;
	// Get matrix
	matrix = extractMatrix(file_name, NODE_COORD_SECTION, nodeRows, nodeColumns);
	// Allocate memory for the array of structs
	node* n = (node*)malloc(nodeRows * sizeof(node));
	if (n == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}
	// Convert to array of struct
	extractNodes(matrix, nodeRows, n);
	// Visualize values for node matrix
	printf("Array of Cities has %d cities \n", nodeRows);
	display(n, nodeRows);

	// Obtain items
	// Calculate amount of rows
	int itemRows = countMatrixRows(file_name, ITEMS_SECTION);
	// Calculate amount of columns
	int itemColumns = 4;
	// Get matrix
	matrix = extractMatrix(file_name, ITEMS_SECTION, itemRows, itemColumns);
	// Allocate memory for the array of structs
	item* i = (item*)malloc(itemRows * sizeof(item));
	if (i == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}
	// Convert to array of struct
	extractItems(matrix, itemRows, i);
	printf("Array of items has %d items \n", itemRows);
	display(i, itemRows);

	// Calculate distances
	distance* d = (distance*)malloc(nodeRows * nodeRows * sizeof(distance));
	if (d == NULL) {
		fprintf(stderr, "Out of Memory");
		exit(0);
	}
	
	euclideanDistance(n, n, d, nodeRows, nodeRows * nodeRows);
	display(d, nodeRows * nodeRows);

	// Calculate Distance Matrix in CUDA
	// Define device pointers
	//float** devDistanceMatrix;
	//int** devNodeMatrix;
	//cudaMalloc((void**)&devDistanceMatrix, nodeRows * nodeRows * sizeof(float*));
	//cudaMalloc((void**)&devNodeMatrix, sizeof(int*) * nodeRows + sizeof(int) * nodeRows * nodeColumns);
	//cudaMemcpy(devNodeMatrix, nodeMatrix, sizeof(int*) * nodeRows + sizeof(int) * nodeRows * nodeColumns, cudaMemcpyHostToDevice);

	// TODO: Implement CUDA Calls

	//cudaFree(devNodeMatrix);
	//cudaFree(devDistanceMatrix);
	free(matrix);
	free(i);
	free(n);
	free(d);
	free(fp);

	// End Execution
	return 0;	
}