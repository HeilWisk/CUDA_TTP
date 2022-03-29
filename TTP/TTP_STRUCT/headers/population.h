// DEFINES: Population as an array of tours
struct population
{
	tour *tours;
};

void initializePopulation(population& initial_population, tour& initial_tour, distance* distances, const int population_size, const int node_quantity)
{
	node temp;

	//Allocate memory for the TOURS
	initial_population.tours = (tour*)malloc(population_size * sizeof(tour));
	if (initial_population.tours == NULL) {
		printf("Unable to allocate memory for nodes");
		return;
	}	

	// Set the tours
	for (int i = 0; i < population_size; ++i)
	{
		//Allocate memory for the nodes on tours
		initial_population.tours[i].nodes = (node*)malloc(node_quantity * sizeof(node));
		if (initial_population.tours[i].nodes == NULL) {
			printf("Unable to allocate memory for nodes");
			return;
		}

		initial_population.tours[i].nodes[0].id = initial_tour.nodes[0].id;
		initial_population.tours[i].nodes[0].x = initial_tour.nodes[0].x;
		initial_population.tours[i].nodes[0].y = initial_tour.nodes[0].y;

		for (int j = 1; j < node_quantity; ++j)
		{
			int random_position = 1 + (rand() % (node_quantity - 1));			
			temp = initial_tour.nodes[j];
			initial_population.tours[i].nodes[j] = initial_tour.nodes[random_position];
			initial_population.tours[i].nodes[random_position] = temp;			
		}		
		evaluateTour(initial_population.tours[i], distances, node_quantity);
	}
}

void printPopulation(population& population, const int population_size, const int node_quantity)
{
	for (int i = 0; i < population_size; ++i)
	{
		printf("Individual %d\n", i);
		printf("> Fitness: %f\n", population.tours[i].fitness);
		printf("> Total Distance: %f\n", population.tours[i].total_distance);
		printf("> Nodes:\n");
		for (int j = 0; j < node_quantity; ++j)
		{
			printf("%d\n", population.tours[i].nodes[j].id);
		}
		printf("\n");
	}
}