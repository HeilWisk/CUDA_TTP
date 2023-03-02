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
		bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, " Profit: %f | Revenue: %f | Time: %f | Distance: %f", pop.tours[i].fitness, pop.tours[i].profit, pop.tours[i].time, pop.tours[i].total_distance);
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
		bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, " Profit: %f | Revenue: %f | Time: %f | Distance: %f", pop.tours[i].fitness, pop.tours[i].profit, pop.tours[i].time, pop.tours[i].total_distance);
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
		bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, " Profit: %f | Revenue: %f | Time: %f | Distance: %f", parents[i].fitness, parents[i].profit, parents[i].time, parents[i].total_distance);
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

	bufferPos = snprintf(bufferWrite, sizeof(char) * WRITE_BUFFER, "");
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
	bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "| %f | %f | %f | %f | %d", fittest.fitness, fittest.profit, fittest.time, fittest.total_distance, generation);
	fprintf(fp, "%s\n", bufferWrite);

	fclose(fp);
	return 1;
}

int saveStatistics(char* name, bool isCuda, int fileNumber, int iteration, double runtimeInitialize, double runtimeSelection, double runtimeCrossover, double runtimeLocalSearch)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];
	int bufferPos;

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_GPU, name, fileNumber);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_CPU, name, fileNumber);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	bufferPos = snprintf(bufferWrite, sizeof(char) * WRITE_BUFFER, "");	
	bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "%d|%f|%f|%f|%f", iteration, runtimeInitialize, runtimeSelection, runtimeCrossover, runtimeLocalSearch);
	fprintf(fp, "%s\n", bufferWrite);

	fclose(fp);
	return 1;
}

int saveGlobalStatistics(char* name, bool isCuda, int fileNumber, double runtimeExecution)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];
	int bufferPos;

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, GLOBALSTATS_FILE_NAME_GPU, name);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, GLOBALSTATS_FILE_NAME_CPU, name);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	bufferPos = snprintf(bufferWrite, sizeof(char) * WRITE_BUFFER, "");
	bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * WRITE_BUFFER) - bufferPos, "%d|%f", fileNumber, runtimeExecution);
	fprintf(fp, "%s\n", bufferWrite);

	fclose(fp);
	return 1;
}

int createStatisticsFile(char* name, bool generateGPUFile, bool generateCPUFile, int fileNumber)
{
	FILE* fp_cpu;
	FILE* fp_gpu;
	char bufferName[NAME_BUFFER];

	if (generateCPUFile)
	{
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_CPU, name, fileNumber);

		fp_cpu = fopen(bufferName, "w");
		if (fp_cpu == NULL)
		{
			printf("No fue posible crear el archivo");
			return 0;
		}

		fprintf(fp_cpu, "EVOLUTION|INITIALPOP TIME|SELECTION TIME|CROSSOVER TIME|LOCALSEARCH TIME\n");
		fclose(fp_cpu);
	}

	if (generateGPUFile)
	{
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, STATISTICS_FILE_NAME_GPU, name, fileNumber);

		fp_gpu = fopen(bufferName, "w");
		if (fp_gpu == NULL)
		{
			printf("No fue posible crear el archivo");
			return 0;
		}

		fprintf(fp_gpu, "EVOLUTION|INITIALPOP TIME|SELECTION TIME|CROSSOVER TIME|LOCALSEARCH TIME\n");
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

		fprintf(fp_cpu, "SOLUTION|PROFIT|REVENUE|TIME|DISTANCE|ITERATION\n");
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

		fprintf(fp_gpu, "SOLUTION|PROFIT|REVENUE|TIME|DISTANCE|ITERATION\n");
		fclose(fp_gpu);
	}
	return 1;
}

int createGlobalStatsFile(char* name, bool generateGPUFile, bool generateCPUFile)
{
	FILE* fp_cpu;
	FILE* fp_gpu;
	char bufferName[NAME_BUFFER];

	if (generateCPUFile)
	{
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, GLOBALSTATS_FILE_NAME_CPU, name);

		fp_cpu = fopen(bufferName, "w");
		if (fp_cpu == NULL)
		{
			printf("No fue posible crear el archivo");
			return 0;
		}

		fprintf(fp_cpu, "EXECUTION|TOTAL TIME\n");
		fclose(fp_cpu);
	}

	if (generateGPUFile)
	{
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, GLOBALSTATS_FILE_NAME_GPU, name);

		fp_gpu = fopen(bufferName, "w");
		if (fp_gpu == NULL)
		{
			printf("No fue posible crear el archivo");
			return 0;
		}

		fprintf(fp_gpu, "EXECUTION|TOTAL TIME\n");
		fclose(fp_gpu);
	}
	return 1;
}

#endif // !OUTPUT_FUNC_H