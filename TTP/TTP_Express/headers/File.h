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

int saveInitialPopulation(char* name, population& pop, parameters& problem, bool isCuda, int fileNumber)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];
	int bufferPos;

	if(isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_CUDA_%s_%d.txt", name, fileNumber);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_%s_%d.txt", name, fileNumber);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	fprintf(fp, "\n");
	fprintf(fp, "INITIAL POPULATION FOR %s", name);
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

int saveOffspring(char* name, population& pop, parameters& problem, int generation, bool isCuda, int fileNumber)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];
	int bufferPos;

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_CUDA_%s_%d.txt", name, fileNumber);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_%s_%d.txt", name, fileNumber);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	fprintf(fp, "\n");
	fprintf(fp, "POPULATION FOR GENERATION %d\n", generation);

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

int saveParents(char* name, tour* parents, parameters& problem, int generation, bool isCuda, int fileNumber)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];
	int bufferPos;

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_CUDA_%s_%d.txt", name, fileNumber);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_%s_%d.txt", name, fileNumber);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	fprintf(fp, "\n");
	fprintf(fp, "PARENTS FOR GENERATION %d\n", generation);

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
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_CUDA_%s_%d.txt", name, fileNumber);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_%s_%d.txt", name, fileNumber);

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

int saveRuntime(char* name, bool isCuda, int fileNumber, const char* kernelName, float runtime, int generation)
{
	FILE* fp;
	char bufferName[NAME_BUFFER];
	char bufferWrite[WRITE_BUFFER];

	if (isCuda)
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_STATISTICS_CUDA_%s_%d.txt", name, fileNumber);
	else
		snprintf(bufferName, sizeof(char) * NAME_BUFFER, ".\\output\\output_STATISTICS_%s_%d.txt", name, fileNumber);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	fprintf(fp, "\n");
	snprintf(bufferWrite, sizeof(char) * WRITE_BUFFER, "Runtime of Kernel %s in generation %d: %f ms", kernelName, generation, runtime);
	fprintf(fp, "%s\n", bufferWrite);

	fclose(fp);
	return 1;
}

#endif // !OUTPUT_FUNC_H