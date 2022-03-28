// DEFINES: Tour Data Type (An array of nodes with certain attributes: Fitness, Distance and an array of item)
struct tour
{
	node nodes[DIMENSION];
	float fitness;
	//TODO: Evaluate how it works with float to try to change for the struct distance
	float distance;
	item items[ITEM_QTY];

	__host__ __device__ tour()
	{
		for (int n = 0; n < DIMENSION; n++)
		{
			nodes[n] = node();
		}

		for (int i = 0; i < ITEM_QTY; i++)
		{
			items[i] = item();
		}

		fitness = 0;
		distance = 0;
	}
};

__host__ __device__ void evaluateTour(tour& tour, const float* cost_table)
{
	tour.distance = 0;
	for (int i = 0; i < DIMENSION; ++i)
	{
		if (i < DIMENSION - 1)
		{
			tour.distance += cost_table[(tour.nodes[i].id) * DIMENSION + (tour.nodes[i + 1]).id];
		}
		else
		{
			tour.distance += cost_table[(tour.nodes[i].id) * DIMENSION + (tour.nodes[0]).id];
		}

		if (tour.distance != 0)
			tour.fitness = 1 / tour.distance;
		else
			tour.fitness = 0;
	}
}

void initializeRandomTour(tour &tour)
{
	// Only randomizes the tail of the tour
	// this is because every tour stars in the same node
	tour.nodes[0] = node(0, 0, 0);
	for (int i = 1; i < DIMENSION; ++i)
	{
		int random_x = rand() % MAX_COORD;
		int random_y = rand() % MAX_COORD;
		tour.nodes[i] = node(i, random_x, random_y);
	}
}

__host__ __device__ void printTour(const tour& tour)
{
	for (int i = 0; i < DIMENSION; i++)
	{
		printf("%d ", tour.nodes[i].id + 1);
	}
}

