// Definition for an item

//DEFINES: Item Data Type
struct item {
	int id;
	float weight;
	float value;
	int node;
	float pw_ratio;

	__host__ __device__ item()
	{
		id = -1;
		weight = -1;
		value = -1;
		node = -1;
		pw_ratio = value / weight;
	}

	__host__ __device__ item(int id_item, float w, float v, int node_id)
	{
		id = id_item;
		weight = w;
		value = v;
		node = node_id;
		pw_ratio = value / weight;
	}

	__host__ __device__ item& operator=(const item& var)
	{
		id = var.id;
		weight = var.weight;
		value = var.value;
		node = var.node;
		pw_ratio = var.pw_ratio;
		return *this;
	}

	__host__ __device__ bool operator==(const item& itm)
		const
	{
		return(id == itm.id && weight == itm.weight && value == itm.value && node == itm.node && pw_ratio == itm.pw_ratio);
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
		i[s].pw_ratio = (float)matrix[s][1] / (float)matrix[s][2];
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

