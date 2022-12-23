// Definition for a struct that contains the problem parameters

// DEFINES: Parameters Data Type
struct parameters {
	char name[100];
	double knapsack_capacity;
	int cities_amount;
	int items_amount;
	double min_speed;
	double max_speed;
	double renting_ratio;
	double items_per_city;
	node cities[CITIES];
	item items[ITEMS];
};

