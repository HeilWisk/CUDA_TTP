// Definition for an item

//DEFINES: Item Data Type
struct item {
	int id;
	float weight;
	float value;
	int node;
	int taken;

	__host__ __device__ item()
	{
		id = -1;
		weight = -1;
		value = -1;
		node = -1;
		taken = 0;
	}

	__host__ __device__ item(int id_item, float w, float v, int node_id, int t)
	{
		id = id_item;
		weight = w;
		value = v;
		node = node_id;
		taken = t;
	}

	__host__ __device__ item& operator=(const item& var)
	{
		id = var.id;
		weight = var.weight;
		value = var.value;
		node = var.node;
		taken = var.taken;
		return *this;
	}
};

/// <summary>
/// Function to convert the extracted matrix into an array of item structs
/// </summary>
/// <param name="matrix">- Matrix to extract</param>
/// <param name="rows">- Amount of rows to extract</param>
/// <param name="i">- Pointer to array of item structs</param>
void extractItems(int** matrix, int rows, item* i) {
	for (int s = 0; s < rows; s++) {
		i[s].id = matrix[s][0];
		i[s].value = (float)matrix[s][1];
		i[s].weight = (float)matrix[s][2];
		i[s].node = matrix[s][3];
		i[s].taken = 0;
	}
}

/// <summary>
/// Display the item array
/// </summary>
/// <param name="c">- Item array</param>
/// <param name="size">- Size of the array</param>
void displayItems(item* c, int size) {
	printf("****************************************************************************************\n");
	printf("ITEMS:		%d\n", size);
	printf("****************************************************************************************\n");
	printf("ID	X		Y		LOC\n");
	for (int i = 0; i < size; i++) {
		printf("%d	%f	%f	%d\n", c[i].id, c[i].value, c[i].weight, c[i].node);
	}
	printf("****************************************************************************************\n");
	printf("\n");
}

