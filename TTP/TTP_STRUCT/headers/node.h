// Definition for a node

// DEFINES: Node Data Type
struct node {
	int id;
	double x;
	double y;
	int item_qty;
	item *items;

	__host__ __device__ node() 
	{ 
		id = -1; 
		x = -1; 
		y = -1;
		item_qty = 0;
	}

	__host__ __device__ node(int id_node, double x_coordinate, double y_coordinate)
	{ 
		id = id_node;
		x = x_coordinate;
		y = y_coordinate;
	}

	__host__ __device__ node& operator=(const node& var)
	{
		id = var.id;
		x = var.x;
		y = var.y;
		item_qty = var.item_qty;
		return *this;
	}

	__host__ __device__ bool operator==(const node& var) const 
	{
		return(id == var.id && x == var.x && y == var.y);
	}
	
};

__host__ __device__ double distanceBetweenNodes(const node& src_node, const node& dst_node)
{
	double x_distance = pow(dst_node.x - src_node.x, 2);
	double y_distance = pow(dst_node.y - src_node.y, 2);
	return sqrt(x_distance + y_distance);
}

/// <summary>
/// Display the node array
/// </summary>
/// <param name="c">- Node array</param>
/// <param name="size">- Size of the array</param>
void displayNodes(node* c, int size) {
	printf("****************************************************************************************\n");
	printf("NODES (CITIES):		%d\n", size);
	printf("****************************************************************************************\n");
	printf("ID	X		Y		ITEMS\n");
	for (int i = 0; i < size; i++) {
		printf("%d	%f	%f", c[i].id, c[i].x, c[i].y);
		if (c[i].item_qty > 0)
		{
			for (int j = 0; j < c[i].item_qty; j++)
			{
				printf("	> %d", c[i].items[j].id);
			}
			printf("\n");
		}
		else
			printf("\n");
	}
	printf("****************************************************************************************\n");
	printf("\n");
}

/// <summary>
/// Function to convert the extracted matrix into an array of node structs
/// </summary>
/// <param name="matrix">- Matrix to extract</param>
/// <param name="rows">- Amount of rows to extract</param>
/// <param name="c">- Pointer to array of nodes structs</param>
void extractNodes(int** matrix, int rows, node* c) {
	for (int i = 0; i < rows; i++) {
		c[i].id = matrix[i][0];
		c[i].x = (float)matrix[i][1];
		c[i].y = (float)matrix[i][2];
	}
}

/// <summary>
/// Function to convert the extracted matrix into an array of item structs
/// </summary>
/// <param name="matrix">- Matrix to extract</param>
/// <param name="rows">- Amount of rows to extract</param>
/// <param name="tour">- Tour to assign the extracted item</param>
void extractItems(int** matrix, int rows, node& node) {
	for (int s = 0; s < rows; s++) {
		node.items[s].id = matrix[s][0];
		node.items[s].value = (float)matrix[s][1];
		node.items[s].weight = (float)matrix[s][2];
		node.items[s].node = matrix[s][3];
	}
}

/// <summary>
/// Function to assign every item to his corresponding node
/// </summary>
/// <param name="items">- Array of items to asign</param>
/// <param name="item_quantity">- Amount of items to asign</param>
/// <param name="nodes">- Nodes to assign the extracted item</param>
/// <param name="node_quantity">- Amount of nodes to be asigned</param>
void assignItems(item* items, int item_quantity, node* nodes, int node_quantity) 
{
	int node_index;
	int amount_items_per_node;
	// Loop through the node array
	for (int n = 0; n < node_quantity; n++)
	{
		// Initialize the amount of items per node with 0
		nodes[n].item_qty = 0;

		// Count the amount of items per node to calculate memory allocation
		amount_items_per_node = 0;
		for (int s = 0; s < item_quantity; s++)
		{
			if (items[s].node == nodes[n].id)
			{
				amount_items_per_node += amount_items_per_node + 1;
			}
		}		

		// Validate if the given node has asigned items
		if (amount_items_per_node > 0)
		{
			// Allocate memory for the item array
			nodes[n].items = (item*)malloc(amount_items_per_node * sizeof(item));
			if (nodes[n].items == NULL) {
				printf("Unable to allocate memory for nodes");
				return;
			}

			node_index = 0;
			for (int s = 0; s < item_quantity; s++)
			{
				if (items[s].node == nodes[n].id)
				{
					nodes[n].items[node_index].id = items[s].id;
					nodes[n].items[node_index].value = items[s].value;
					nodes[n].items[node_index].weight = items[s].weight;
					nodes[n].items[node_index].node = items[s].node;
					nodes[n].items[node_index].taken = items[s].taken;
					nodes[n].item_qty++;
					node_index++;
				}
			}
		}
	}
}