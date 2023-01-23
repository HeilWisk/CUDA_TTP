// DEFINES: Tour Data Type (An array of nodes with certain attributes: Fitness, Distance and an array of item)
struct tour
{
	double fitness;
	double total_distance;
	double profit;
	double time;
	node nodes[CITIES+1];
	item item_picks[ITEMS];

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	__host__ __device__ tour()
	{
		for (int i = 0; i < CITIES + 1; ++i)
		{
			nodes[i] = node(-1, -1, -1);
		}

		for (int j = 0; j < ITEMS; j++)
		{
			item_picks[j] = item();
		}

		fitness = 0;
		total_distance = 0;
		time = 0;
		profit = 0;
	}

	/// <summary>
	/// Overload to the equals (=) operator
	/// </summary>
	/// <param name="t"></param>
	/// <returns></returns>
	__host__ __device__ tour& operator=(const tour& t)
	{
		for (int i = 0; i < CITIES + 1; ++i)
		{
			nodes[i] = t.nodes[i];
		}

		for (int j = 0; j < ITEMS; j++)
		{
			item_picks[j] = t.item_picks[j];
		}

		fitness = t.fitness;
		total_distance = t.total_distance;
		time = t.time;
		profit = t.profit;
		return *this;
	}

	/// <summary>
	/// Overload of the isequal (==) operator
	/// </summary>
	/// <param name="t"></param>
	/// <returns></returns>
	__host__ __device__ bool operator==(tour& t)
	{
		for (int i = 0; i < CITIES + 1; ++i)
		{
			if (nodes[i].x != t.nodes[i].x || nodes[i].y != t.nodes[i].y || nodes[i].items != t.nodes[i].items)
			{
				return false;
			}
		}
		return true;
	}

	
};

/// <summary>
/// Evaluate the fitness of a tour
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
/// <param name="individual"></param>
/// <param name="distance_table"></param>
/// <param name="problem_params"></param>
/// <returns></returns>
__host__ __device__ void evaluateTour(tour& individual, parameters problem_params)
{
	//tour result = individual;
	double carrying = 0.0;
	double profit = 0.0;
	double time = 0.0;
	double revenue = 0.0;
	double velocity = 0.0;
	double distance = 0.0;

	// Obtain the total weight of the items and the quantity of items
	double total_weight = 0.0;
	int item_count = 0;
	for (int i = 0; i < CITIES + 1; ++i)
	{
		for (int j = 0; j < ITEMS; ++j)
		{
			if (individual.nodes[i].items[j].id > 0)
			{
				//individual.item_picks[item_count] = individual.nodes[i].items[j];
				individual.item_picks[(individual.nodes[i].items[j].id - 1)].pickup = individual.nodes[i].items[j].pickup;
				++item_count;
				if (individual.nodes[i].items[j].pickup == 1)
				{
					total_weight += individual.nodes[i].items[j].weight;
				}
			}
		}
	}

	if (total_weight > problem_params.knapsack_capacity)
	{
		time += total_weight;
	}

	for (int y = 0; y < CITIES; ++y)
	{
		node index = individual.nodes[y];

		for (int z = 0; z < ITEMS; ++z)
		{
			if (individual.item_picks[z].id > 0 && individual.item_picks[z].node == index.id && individual.item_picks[z].pickup == 1)
			{
				carrying += individual.item_picks[z].weight;
				profit += individual.item_picks[z].value;
			}
		}

		velocity = problem_params.max_speed - carrying * (problem_params.max_speed - problem_params.min_speed) / problem_params.knapsack_capacity;
		if (y + 1 == CITIES)
			distance = distanceBetweenNodes(individual.nodes[y], individual.nodes[0]);
		else
			distance = distanceBetweenNodes(individual.nodes[y], individual.nodes[y + 1]);

		time += distance / velocity;
		revenue = profit - (problem_params.renting_ratio * time);
		distance += distance;
		//individual.total_distance += distance;
		//return result;
	}

	individual.total_distance = distance;
	individual.profit = profit;
	individual.time = time;
	individual.fitness = revenue;
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
	printf("TIME: %f\n", tour.time);
	printf("PROFIT: %f\n", tour.profit);
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

/// <summary>
/// Define the initial tour
/// </summary>
/// <param name="initial_tour"></param>
/// <param name="node_quantity"></param>
/// <param name="nodes"></param>
/// void defineInitialTour(tour& initial_tour, const int node_quantity, node* nodes)
void defineInitialTour(tour& initial_tour, parameters& params, node* nodes, item* items)
{
	// Load data on nodes	
	for (int n = 0; n < params.cities_amount; n++)
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
			initial_tour.nodes[n].items[i].pw_ratio = nodes[n].items[i].value / nodes[n].items[i].weight;
		}
		params.cities[n] = initial_tour.nodes[n];
	}

	// Copy data from node 0 to the last node of the tour because in TTP the thief must return to the origin city (node)
	initial_tour.nodes[CITIES] = initial_tour.nodes[0];

	// Initialize item picks
	for (int a = 0; a < ITEMS; ++a)
	{
		initial_tour.item_picks[a] = items[a];
	}

	initial_tour.fitness = 0;
	initial_tour.total_distance = 0;
	initial_tour.profit = 0;
	initial_tour.time = 0;
}