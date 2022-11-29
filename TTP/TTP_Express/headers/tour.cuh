// DEFINES: Tour Data Type (An array of nodes with certain attributes: Fitness, Distance and an array of item)
struct tour
{
	double fitness;
	double total_distance;
	node nodes[CITIES];

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	__host__ __device__ tour()
	{
		for (int i = 0; i < CITIES; ++i)
		{
			nodes[i] = node(-1, -1, -1);
		}
		fitness = 0;
		total_distance = 0;		
	}

	/// <summary>
	/// Overload to the equals (=) operator
	/// </summary>
	/// <param name="t"></param>
	/// <returns></returns>
	__host__ __device__ tour& operator=(const tour& t)
	{
		for (int i = 0; i < CITIES; ++i)
		{
			nodes[i] = t.nodes[i];
		}
		fitness = t.fitness;
		total_distance = t.total_distance;
		return *this;
	}

	/// <summary>
	/// Overload of the isequal (==) operator
	/// </summary>
	/// <param name="t"></param>
	/// <returns></returns>
	__host__ __device__ bool operator==(tour& t)
	{
		for (int i = 0; i < CITIES; ++i)
		{
			if (nodes[i].x != t.nodes[i].x || nodes[i].y != t.nodes[i].y)
			{
				return false;
			}
		}
		return true;
	}

	
};

/// <summary>
/// 
/// </summary>
/// <param name="tour"></param>
/// <param name="distance_table"></param>
/// <param name="node_quantity"></param>
__host__ __device__ void evaluateTour(tour& tour, const distance* distance_table)
{
	tour.total_distance = 0;
	for (int i = 0; i < CITIES; ++i)
	{
		for (int k = 0; k < CITIES * CITIES; ++k)
		{
			if (i < CITIES - 1)
			{
				if ((distance_table[k].source == tour.nodes[i].id) && (distance_table[k].destiny == tour.nodes[i + 1].id))
				{
					tour.total_distance += distance_table[k].value;
				}
			}
			else
			{
				if ((distance_table[k].source == tour.nodes[i].id) && (distance_table[k].destiny == tour.nodes[0].id))
				{
					tour.total_distance += distance_table[k].value;
				}
			}
		}

		// Calculate the fitness
		if (tour.total_distance != 0)
			tour.fitness = 1 / tour.total_distance;
		else
			tour.fitness = 0;
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="tour"></param>
/// <param name="node_quantity"></param>
void initializeRandomTour(tour& tour, const int node_quantity)
{
	// Only randomizes the tail of the tour
	// this is because every tour stars in the same node
	tour.nodes[0] = node(0, 0, 0);
	for (int i = 1; i < node_quantity; ++i)
	{
		double random_x = rand() % MAX_COORD;
		double random_y = rand() % MAX_COORD;
		tour.nodes[i] = node(i, random_x, random_y);
	}
}

__host__ __device__ void printTour(const tour& tour, const int node_quantity)
{
	printf("TOUR INFORMATION\n");
	printf("FITNESS: %f\n", tour.fitness);
	printf("NODES:\n");
	for (int i = 0; i < node_quantity; ++i)
	{
		printf("NODE[%d]	ID: %d\n", i, tour.nodes[i].id);
	}

	printf("TOTAL DISTANCE: %f\n", tour.total_distance);
	printf("\n");
}

/// <summary>
/// Function to convert the extracted matrix into an array of node structs
/// </summary>
/// <param name="matrix">- Matrix to extract</param>
/// <param name="rows">- Amount of rows to extract</param>
/// <param name="tour">- Tour to assign the extracted nodes</param>
void extractNodes(int** matrix, int rows, tour& tour)
{
	for (int i = 0; i < rows; i++) {
		tour.nodes[i].id = matrix[i][0];
		tour.nodes[i].x = matrix[i][1];
		tour.nodes[i].y = matrix[i][2];
	}
}

void defineInitialTour(tour& initial_tour, const int node_quantity, node* nodes)
{
	//Load data on nodes
	for (int n = 0; n < node_quantity; n++)
	{
		initial_tour.nodes[n] = node();
		initial_tour.nodes[n].id = nodes[n].id;
		initial_tour.nodes[n].x = nodes[n].x;
		initial_tour.nodes[n].y = nodes[n].y;

		//Load data on items
		for (int i = 0; i < ITEMS; i++)
		{
			initial_tour.nodes[n].items[i] = item();
			initial_tour.nodes[n].items[i].id = nodes[n].items[i].id;
			initial_tour.nodes[n].items[i].value = nodes[n].items[i].value;
			initial_tour.nodes[n].items[i].weight = nodes[n].items[i].weight;
			initial_tour.nodes[n].items[i].node = nodes[n].items[i].node;
			initial_tour.nodes[n].items[i].taken = nodes[n].items[i].taken;
		}
	}

	initial_tour.fitness = 0;
	initial_tour.total_distance = 0;
}