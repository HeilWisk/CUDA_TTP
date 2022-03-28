// DEFINES: Population as an array of tours
struct population
{
	tour *tours;
};

void initializePopulation(population& initial_population, tour& initial_tour, float* cost_table, const int population_size, const int node_quantity)
{
	node temp;
	initial_population.tours[0] = initial_tour;
	for (int i = 1; i < population_size; ++i)
	{
		for (int j = 1; j < node_quantity; ++j)
		{
			int random_position = 1 + (rand() % (node_quantity - 1));
			temp = initial_tour.nodes[j];
			initial_tour.nodes[j] = initial_tour.nodes[random_position];
			initial_tour.nodes[random_position] = temp;
		}
		initial_population.tours[i] = initial_tour;
		evaluateTour(initial_population.tours[i], cost_table, node_quantity);
	}
}

void printPopulation(population& population, const int population_size, const int node_quantity)
{
	for (int i = 0; i < population_size; ++i)
	{
		printf("Individual %d\n", i);
		printf("> Fitness: %d\n", population.tours[i].fitness);
		printf("> Nodes:\n");
		for (int j = 0; j < node_quantity; ++j)
		{
			printf("%d\n", population.tours[i].nodes[j].id);
		}
		printf("\n");
	}
}