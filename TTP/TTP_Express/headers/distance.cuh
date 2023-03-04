// Definition for distance

//DEFINES: distance data type
struct distance {
	int source;
	int destiny;
	float value;

	__host__ __device__ distance()
	{
		source = -1;
		destiny = -1;
		value = -1;
	}

	__host__ __device__ distance(int source_id, int destiny_id, float distance_value)
	{
		source = source_id;
		destiny = destiny_id;
		value = distance_value;
	}
};

/// <summary>
/// Display the distances array
/// </summary>
/// <param name="d">- Distances array</param>
/// <param name="size">- Size of the array</param>
void displayDistance(distance* d, int size) {
	printf("****************************************************************************************\n");
	printf("DISTANCES\n");
	printf("****************************************************************************************\n");
	printf("SOURCE	DESTINY	DISTANCE\n");
	for (int i = 0; i < size; i++) {
		printf("%d	%d	%f\n", d[i].source, d[i].destiny, d[i].value);
	}
	printf("****************************************************************************************\n");
	printf("\n");
}
