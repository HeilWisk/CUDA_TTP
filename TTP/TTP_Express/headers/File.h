#ifndef OUTPUT_FUNC_H
#define OUTPUT_FUNC_H

/// <summary>
/// Create output file
/// </summary>
/// <param name="name">Name of the problem</param>
/// <returns></returns>
int createFile(char* name, int fileNumber)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];

	snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_%s_%d.txt", name, fileNumber);

	fp = fopen(bufferName, "w");
	if (fp == NULL)
	{
		printf("No fue posible crear el archivo");
		return 0;
	}

	fclose(fp);
	return 1;
}

int saveInitialPopulation(char* name, population& pop, parameters& problem, bool isCuda, int fileNumber, double runtime)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];
	int bufferPos;

	if(isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_GPU, fileNumber, name);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_CPU, fileNumber, name);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	fprintf(fp, "\n");
	fprintf(fp, "INITIAL POPULATION FOR %s", name);
	fprintf(fp, "\n");
	fprintf(fp, "RUNTIME: %f ms", runtime);
	fprintf(fp, "\n");

	for (int i = 0; i < TOURS; ++i)
	{
		bufferPos = snprintf(bufferWrite, sizeof(char) * WRITE_BUFFER, "Individual %d: ", i);
		for (int j = 0; j < problem.cities_amount + 1; ++j)
		{
			if (pop.tours[i].nodes[j].id > 0)
			{
				if (j > 0)
				{
					bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, ", ");
				}					
				bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "%d", pop.tours[i].nodes[j].id);

				for (int k = 0; k < problem.items_per_city; ++k)
				{
					if (pop.tours[i].nodes[j].items[k].id > 0)
					{
						if (k > 0)
						{
							bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, ", ");
						}
						bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "[%d]", pop.tours[i].nodes[j].items[k].pickup);
					}
				}
			}				
		}
		bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, " Profit: %f - Revenue: %f - Time: %f - Distance: %f", pop.tours[i].fitness, pop.tours[i].profit, pop.tours[i].time, pop.tours[i].total_distance);
		fprintf(fp, "%s\n", bufferWrite);
	}

	fclose(fp);
	return 1;
}

int saveOffspring(char* name, population& pop, parameters& problem, int generation, bool isCuda, int fileNumber, double crossoverRuntime, double localSearchRuntime)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];
	int bufferPos;

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_GPU, fileNumber, name);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_CPU, fileNumber, name);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	fprintf(fp, "\n");
	fprintf(fp, "POPULATION FOR GENERATION %d", generation);
	fprintf(fp, "\n");
	fprintf(fp, "CROSSOVER RUNTIME: %f ms", crossoverRuntime);
	fprintf(fp, "\n");
	fprintf(fp, "LOCAL SEARCH RUNTIME: %f ms", localSearchRuntime);
	fprintf(fp, "\n");

	for (int i = 0; i < TOURS; ++i)
	{
		bufferPos = snprintf(bufferWrite, sizeof(char) * WRITE_BUFFER, "Individual %d: ", i);
		for (int j = 0; j < problem.cities_amount + 1; ++j)
		{
			if (pop.tours[i].nodes[j].id > 0)
			{
				if (j > 0)
				{
					bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, ", ");
				}
				bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "%d", pop.tours[i].nodes[j].id);

				for (int k = 0; k < problem.items_per_city; ++k)
				{
					if (pop.tours[i].nodes[j].items[k].id > 0)
					{
						if (k > 0)
						{
							bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, ", ");
						}
						bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "[%d]", pop.tours[i].nodes[j].items[k].pickup);
					}
				}
			}
		}
		bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, " Profit: %f - Revenue: %f - Time: %f - Distance: %f", pop.tours[i].fitness, pop.tours[i].profit, pop.tours[i].time, pop.tours[i].total_distance);
		fprintf(fp, "%s\n", bufferWrite);
	}

	fclose(fp);
	return 1;
}

int saveParents(char* name, tour* parents, parameters& problem, int generation, bool isCuda, int fileNumber, double runtime)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];
	int bufferPos;

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_GPU, fileNumber, name);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_CPU, fileNumber, name);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	fprintf(fp, "\n");
	fprintf(fp, "PARENTS FOR GENERATION %d", generation);
	fprintf(fp, "\n");
	fprintf(fp, "RUNTIME: %f ms", runtime);
	fprintf(fp, "\n");

	for (int i = 0; i < SELECTED_PARENTS; ++i)
	{
		bufferPos = snprintf(bufferWrite, sizeof(char) * WRITE_BUFFER, "Parent %d: ", i);
		for (int j = 0; j < problem.cities_amount + 1; ++j)
		{
			if (parents[i].nodes[j].id > 0)
			{
				if (j > 0)
				{
					bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, ", ");
				}
				bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "%d", parents[i].nodes[j].id);

				for (int k = 0; k < problem.items_per_city; ++k)
				{
					if (parents[i].nodes[j].items[k].id > 0)
					{
						if (k > 0)
						{
							bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, ", ");
						}
						bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "[%d]", parents[i].nodes[j].items[k].pickup);
					}
				}
			}
		}
		bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, " Profit: %f - Revenue: %f - Time: %f - Distance: %f", parents[i].fitness, parents[i].profit, parents[i].time, parents[i].total_distance);
		fprintf(fp, "%s\n", bufferWrite);
	}

	fclose(fp);
	return 1;
}

int saveFittest(char* name, tour fittest, parameters& problem, int generation, bool isCuda, int fileNumber)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];
	int bufferPos;

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_GPU, fileNumber, name);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_CPU, fileNumber, name);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	fprintf(fp, "\n");
	fprintf(fp, "FITTEST OF GENERATION %d\n", generation);
	bufferPos = snprintf(bufferWrite, sizeof(char) * WRITE_BUFFER, "Fittest: ");
	for (int j = 0; j < problem.cities_amount + 1; ++j)
	{
		if (fittest.nodes[j].id > 0)
		{
			if (j > 0)
			{
				bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, ", ");
			}
			bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "%d", fittest.nodes[j].id);

			for (int k = 0; k < problem.items_per_city; ++k)
			{
				if (fittest.nodes[j].items[k].id > 0)
				{
					if (k > 0)
					{
						bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, ", ");
					}
					bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "[%d]", fittest.nodes[j].items[k].pickup);
				}
			}
		}
	}
	bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, " Profit: %f - Revenue: %f - Time: %f - Distance: %f", fittest.fitness, fittest.profit, fittest.time, fittest.total_distance);
	fprintf(fp, "%s\n", bufferWrite);

	fclose(fp);
	return 1;
}

int saveStatistics(char* name, bool isCuda, int fileNumber, double runtimeInitialize, double meanSelection, double meanCrossover, double meanLocalSearch, double medianSelection, double medianCrossover, double medianLocalSearch, double modeSelection, double modeCrossover, double modeLocalSearch, double sdSelection, double sdCrossover, double sdLocalSearch, double meanSolution, double medianSolution, double modeSolution, double sdSolution, double runtimeExecution)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_GPU, name);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_CPU, name);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}
	fprintf(fp, "\n");
	fprintf(fp, "EXECUTION %d", fileNumber);
	fprintf(fp, "\n");
	fprintf(fp, "INITIALIZE POPULATION: %f ms", runtimeInitialize);
	fprintf(fp, "\n");
	fprintf(fp, "MEAN: SELECTION %f ms - CROSSOVER %f ms - LOCAL SEARCH %f ms - SOLUTION %f", meanSelection, meanCrossover, meanLocalSearch, meanSolution);
	fprintf(fp, "\n");
	fprintf(fp, "MEDIAN: SELECTION %f ms - CROSSOVER %f ms - LOCAL SEARCH %f ms - SOLUTION %f", medianSelection, medianCrossover, medianLocalSearch, medianSolution);
	fprintf(fp, "\n");
	fprintf(fp, "MODE: SELECTION %f ms - CROSSOVER %f ms - LOCAL SEARCH %f ms - SOLUTION %f", modeSelection, modeCrossover, modeLocalSearch, modeSolution);
	fprintf(fp, "\n");
	fprintf(fp, "STANDARD DEVIATION: SELECTION %f ms - CROSSOVER %f ms - LOCAL SEARCH %f ms - SOLUTION %f", sdSelection, sdCrossover, sdLocalSearch, sdSolution);
	fprintf(fp, "\n");
	fprintf(fp, "TOTAL RUNTIME: %f ms", runtimeExecution);
	fprintf(fp, "\n");

	fclose(fp);
	return 1;
}

int saveGlobalStatistics(char* name, bool isCuda, double meanInitialize, double meanSelection, double meanCrossover, double meanLocalSearch, double meanTotalRuntime, double medianInitialize, double medianSelection, double medianCrossover, double medianLocalSearch, double medianTotalRuntime, double modeInitialize, double modeSelection, double modeCrossover, double modeLocalSearch, double modeTotalRuntime, double sdInitialize, double sdSelection, double sdCrossover, double sdLocalSearch, double meanSolution, double medianSolution, double modeSolution, double sdSolution, double sdTotalRuntime)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_GPU, name);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_CPU, name);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}
	fprintf(fp, "\n");
	fprintf(fp, "\n");
	fprintf(fp, "GLOBAL STATISTICS");
	fprintf(fp, "\n");
	fprintf(fp, "MEAN: INITIALIZE POPULATION %f ms - SELECTION %f ms - CROSSOVER %f ms - LOCAL SEARCH %f ms - SOLUTION %f - TOTAL RUNTIME %f ms", meanInitialize, meanSelection, meanCrossover, meanLocalSearch, meanSolution, meanTotalRuntime);
	fprintf(fp, "\n");
	fprintf(fp, "MEDIAN: INITIALIZE POPULATION %f ms - SELECTION %f ms - CROSSOVER %f ms - LOCAL SEARCH %f ms - SOLUTION %f - TOTAL RUNTIME %f ms", medianInitialize, medianSelection, medianCrossover, medianLocalSearch, medianSolution,  medianTotalRuntime);
	fprintf(fp, "\n");
	fprintf(fp, "MODE: INITIALIZE POPULATION %f ms - SELECTION %f ms - CROSSOVER %f ms - LOCAL SEARCH %f ms - SOLUTION %f - TOTAL RUNTIME %f ms", modeInitialize, modeSelection, modeCrossover, modeLocalSearch, modeSolution, modeTotalRuntime);
	fprintf(fp, "\n");
	fprintf(fp, "STANDARD DEVIATION: INITIALIZE POPULATION %f ms - SELECTION %f ms - CROSSOVER %f ms - LOCAL SEARCH %f ms - SOLUTION %f - TOTAL RUNTIME %f ms", sdInitialize, sdSelection, sdCrossover, sdLocalSearch, sdSolution, sdTotalRuntime);
	fprintf(fp, "\n");

	fclose(fp);
	return 1;
}

int createStatisticsFile(char* name, bool generateGPUFile, bool generateCPUFile)
{
	FILE* fp_cpu;
	FILE* fp_gpu;
	char bufferName[NAME_BUFFER];

	if (generateCPUFile)
	{
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_CPU, name);

		fp_cpu = fopen(bufferName, "w");
		if (fp_cpu == NULL)
		{
			printf("No fue posible crear el archivo");
			return 0;
		}

		fprintf(fp_cpu, "PROBLEM NAME: %s\n", name);
		fprintf(fp_cpu, "EXECUTIONS: %d\n", NUMBER_EXECUTIONS);
		fprintf(fp_cpu, "SOLUTIONS PER GENERATION: %d\n", TOURS);
		fprintf(fp_cpu, "CITIES: %d\n", CITIES);
		fprintf(fp_cpu, "ITEMS: %d\n", ITEMS);
		fprintf(fp_cpu, "ITEMS PER CITY: %d\n", ITEMS_PER_CITY);
		fprintf(fp_cpu, "EVOLUTIONS: %d\n", NUM_EVOLUTIONS);
		fprintf(fp_cpu, "TOURNAMENT SIZE: %d\n", TOURNAMENT_SIZE);
		fprintf(fp_cpu, "PARENTS PER GENERATIONS: %d\n", SELECTED_PARENTS);
		fprintf(fp_cpu, "LOCAL SEARCH PROBABILITY: %f\n", LOCAL_SEARCH_PROBABILITY);
		fprintf(fp_cpu, "\n");

		fclose(fp_cpu);
	}

	if (generateGPUFile)
	{
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_GPU, name);

		fp_gpu = fopen(bufferName, "w");
		if (fp_gpu == NULL)
		{
			printf("No fue posible crear el archivo");
			return 0;
		}

		fprintf(fp_gpu, "PROBLEM NAME: %s\n", name);
		fprintf(fp_gpu, "EXECUTIONS: %d\n", NUMBER_EXECUTIONS);
		fprintf(fp_gpu, "SOLUTIONS PER GENERATION: %d\n", TOURS);
		fprintf(fp_gpu, "CITIES: %d\n", CITIES);
		fprintf(fp_gpu, "ITEMS: %d\n", ITEMS);
		fprintf(fp_gpu, "ITEMS PER CITY: %d\n", ITEMS_PER_CITY);
		fprintf(fp_gpu, "EVOLUTIONS: %d\n", NUM_EVOLUTIONS);
		fprintf(fp_gpu, "TOURNAMENT SIZE: %d\n", TOURNAMENT_SIZE);
		fprintf(fp_gpu, "PARENTS PER GENERATIONS: %d\n", SELECTED_PARENTS);
		fprintf(fp_gpu, "LOCAL SEARCH PROBABILITY: %f\n", LOCAL_SEARCH_PROBABILITY);
		fprintf(fp_gpu, "\n");

		fclose(fp_gpu);
	}
	return 1;
}

int createOutputFile(char* name, bool generateGPUFile, bool generateCPUFile, int fileNumber)
{
	FILE* fp_cpu;
	FILE* fp_gpu;
	char bufferName[NAME_BUFFER];

	if (generateCPUFile)
	{
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_CPU, fileNumber, name);

		fp_cpu = fopen(bufferName, "w");
		if (fp_cpu == NULL)
		{
			printf("No fue posible crear el archivo");
			return 0;
		}

		fprintf(fp_cpu, "PROBLEM NAME: %s\n", name);
		fprintf(fp_cpu, "SOLUTIONS PER GENERATION: %d\n", TOURS);
		fprintf(fp_cpu, "CITIES: %d\n", CITIES);
		fprintf(fp_cpu, "ITEMS: %d\n", ITEMS);
		fprintf(fp_cpu, "ITEMS PER CITY: %d\n", ITEMS_PER_CITY);
		fprintf(fp_cpu, "EVOLUTIONS: %d\n", NUM_EVOLUTIONS);
		fprintf(fp_cpu, "TOURNAMENT SIZE: %d\n", TOURNAMENT_SIZE);
		fprintf(fp_cpu, "PARENTS PER GENERATIONS: %d\n", SELECTED_PARENTS);
		fprintf(fp_cpu, "LOCAL SEARCH PROBABILITY: %f\n", LOCAL_SEARCH_PROBABILITY);
		fprintf(fp_cpu, "\n");

		fclose(fp_cpu);
	}

	if (generateGPUFile)
	{
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, RESULTS_FILE_NAME_GPU, fileNumber, name);

		fp_gpu = fopen(bufferName, "w");
		if (fp_gpu == NULL)
		{
			printf("No fue posible crear el archivo");
			return 0;
		}

		fprintf(fp_gpu, "PROBLEM NAME: %s\n", name);
		fprintf(fp_gpu, "SOLUTIONS PER GENERATION: %d\n", TOURS);
		fprintf(fp_gpu, "CITIES: %d\n", CITIES);
		fprintf(fp_gpu, "ITEMS: %d\n", ITEMS);
		fprintf(fp_gpu, "ITEMS PER CITY: %d\n", ITEMS_PER_CITY);
		fprintf(fp_gpu, "EVOLUTIONS: %d\n", NUM_EVOLUTIONS);
		fprintf(fp_gpu, "TOURNAMENT SIZE: %d\n", TOURNAMENT_SIZE);
		fprintf(fp_gpu, "PARENTS PER GENERATIONS: %d\n", SELECTED_PARENTS);
		fprintf(fp_gpu, "LOCAL SEARCH PROBABILITY: %f\n", LOCAL_SEARCH_PROBABILITY);
		fprintf(fp_gpu, "\n");

		fclose(fp_gpu);
	}
	return 1;
}

#endif // !OUTPUT_FUNC_H