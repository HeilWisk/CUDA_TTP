// Definition for an item

//DEFINES: Item Data Type
struct item {
	int id;
	short weight;
	short value;
	int node;
	int pickup;

	__host__ __device__ item()
	{
		id = -1;
		weight = -1;
		value = -1;
		node = -1;
		pickup = -1;
	}

	__host__ __device__ item(int id_item, int w, int v, int node_id)
	{
		id = id_item;
		weight = w;
		value = v;
		node = node_id;
		pickup = 0;
	}

	__host__ __device__ item& operator=(const item& var)
	{
		id = var.id;
		weight = var.weight;
		value = var.value;
		node = var.node;
		pickup = var.pickup;
		return *this;
	}

	__host__ __device__ bool operator==(const item& itm)
		const
	{
		return(id == itm.id && weight == itm.weight && value == itm.value && node == itm.node);
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
		i[s].value = matrix[s][1];
		i[s].weight = matrix[s][2];
		i[s].node = matrix[s][3];
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
		printf("%d	%d	%d	%d\n", c[i].id, c[i].value, c[i].weight, c[i].node);
	}
	printf("****************************************************************************************\n");
	printf("\n");
}

void randomPickup(item* items)
{
	for (int i = 0; i < ITEMS_PER_CITY; ++i)
	{
		if (items[i].id > 0)
			items[i].pickup = rand() % 2;
	}
}

/// <summary>
/// 
/// </summary>
/// <param name="items"></param>
/// <param name="state"></param>
/// <returns></returns>
__device__ void randomPickup(item* items, curandState* state)
{
	for (int i = 0; i < ITEMS_PER_CITY; ++i)
	{
		if (items[i].id > 0)
			items[i].pickup = curand(state) % 2;
	}
}

