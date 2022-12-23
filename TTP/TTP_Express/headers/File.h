#ifndef OUTPUT_FUNC_H
#define OUTPUT_FUNC_H

/// <summary>
/// Create output file
/// </summary>
/// <param name="name">Name of the problem</param>
/// <returns></returns>
int createFile(char* name)
{
	FILE* fp;
	char bufferName[100];

	snprintf(bufferName, sizeof(char) * 100, ".\\output\\output_%s.txt", name);

	fp = fopen(bufferName, "w");
	if (fp == NULL)
	{
		printf("No fue posible crear el archivo");
		return 0;
	}

	fprintf(fp, "INITIAL POPULATION FOR %s", name);
	fclose(fp);
	return 1;
}

int saveInitialPopulation(char* name, population& pop, parameters& problem)
{
	FILE* fp;
	char bufferName[100];
	char bufferWrite[100];
	int bufferPos;

	snprintf(bufferName, sizeof(char) * 100, ".\\output\\output_%s.txt", name);

	fp = fopen(bufferName, "a");
	if (fp == NULL)
	{
		printf("No fue posible abrir el archivo");
		return 0;
	}

	fprintf(fp, "\n");

	for (int i = 0; i < TOURS; ++i)
	{
		bufferPos = snprintf(bufferWrite, sizeof(char) * 100, "Individual %d: ", i);
		for (int j = 0; j < problem.cities_amount + 1; ++j)
		{
			if (pop.tours[i].nodes[j].id > 0)
			{
				if (j > 0)
				{
					bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * 100) - bufferPos, ", ");
				}					
				bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * 100) - bufferPos, "%d", pop.tours[i].nodes[j].id);

				for (int k = 0; k < problem.items_amount; ++k)
				{
					if (pop.tours[i].nodes[j].items[k].id > 0)
					{
						if (k > 0)
						{
							bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * 100) - bufferPos, ", ");
						}
						bufferPos += snprintf(bufferWrite + bufferPos, (sizeof(char) * 100) - bufferPos, "[%d]", pop.tours[i].nodes[j].items[k].pickup);
					}
				}
			}				
		}
		fprintf(fp, "%s\n", bufferWrite);
	}

	fclose(fp);
	return 1;
}

#endif // !OUTPUT_FUNC_H