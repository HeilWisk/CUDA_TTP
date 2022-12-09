// DEFINES: Population as an array of tours
struct population
{
	tour tours[TOURS];

	__host__ __device__ population& operator=(const population& pop)
	{
		for (int i = 0; i < TOURS; ++i)
		{
			tours[i] = pop.tours[i];
		}
		return *this;
	}
};

/// <summary>
/// Initialize Random Population
/// </summary>
/// <param name="initialPopulation"></param>
/// <param name="initialTour"></param>
/// <param name="distances"></param>
void initializePopulation(population& initialPopulation, tour& initialTour, distance* distances)
{
	initialPopulation.tours[0] = initialTour;
	for (int i = 1; i < TOURS; ++i)
	{
		evaluateTour(initialPopulation.tours[0], distances);
		for (int j = 1; j < CITIES; ++j)
		{
			int randPos = 1 + (rand() % (CITIES - 1));
			node tempNode = initialTour.nodes[j];
			initialTour.nodes[j] = initialTour.nodes[randPos];
			initialTour.nodes[randPos] = tempNode;
		}
		initialPopulation.tours[i] = initialTour;
		evaluateTour(initialPopulation.tours[i], distances);
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="initialPopulation"></param>
/// <param name="initialTour"></param>
/// <param name="problem_params"></param>
void initializePopulation(population& initialPopulation, tour& initialTour, parameters problem_params)
{
	// Generate random pickup for the items in each node od the tour
	for (int i = 0; i < CITIES; ++i)
	{
		randomPickup(initialTour.nodes[i].items);
	}
	initialPopulation.tours[0] = initialTour;	
	evaluateTour(initialPopulation.tours[0], problem_params);

	for (int i = 1; i < TOURS; ++i)
	{
		for (int j = 1; j < CITIES; ++j)
		{
			randomPickup(initialTour.nodes[j].items);
			int randPos = 1 + (rand() % (CITIES - 1));
			node tempNode = initialTour.nodes[j];
			initialTour.nodes[j] = initialTour.nodes[randPos];
			initialTour.nodes[randPos] = tempNode;			
		}
		initialPopulation.tours[i] = initialTour;
		evaluateTour(initialPopulation.tours[i], problem_params);
	}
}

/// <summary>
/// Function to print the randomly generated population
/// </summary>
/// <param name="population"></param>
void printPopulation(population population)
{
	for (int i = 0; i < TOURS; ++i)
	{
		printf("Individual %d\n", i);
		printf("> Fitness: %f\n", population.tours[i].fitness);
		printf("> Profit: %f\n", population.tours[i].profit);
		printf("> Total Distance: %f\n", population.tours[i].total_distance);
		printf("> Time: %f\n", population.tours[i].time);
		printf("> Nodes		>Item ID[PICKED]\n");
		for (int j = 0; j < CITIES + 1; ++j)
		{
			if (population.tours[i].nodes[j].id > 0)
			{
				printf("> %d", population.tours[i].nodes[j].id);

				for (int h = 0; h < ITEMS; h++)
				{
					if(population.tours[i].nodes[j].items[h].id > 0)
						printf("		> %d[%d]", population.tours[i].nodes[j].items[h].id, population.tours[i].nodes[j].items[h].pickup);
				}
				printf("\n");
			}
		}
		printf("\n\n");
	}
}