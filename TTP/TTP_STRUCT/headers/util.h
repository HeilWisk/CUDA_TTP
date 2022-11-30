/// <summary>
/// Function to test memory allocation in CUDA
/// </summary>
/// <param name="initial_population"></param>
/// <param name="population_size"></param>
/// <param name="node_quantity"></param>
/// <param name="item_quantity"></param>
/// <returns></returns>
void testMemoryAllocationCPU(population* initial_population, const int population_size)
{
	printf(" > population->id: %d", initial_population->id);
	for (int p = 0; p < population_size; ++p)
	{
		printf(" > population[%d].id: %d", p, initial_population[p].id);
		printf(" > population[%d].tours:", p);
		printf(" > population[%d].tours->fitness: %f", p, initial_population[p].tours->fitness);
		printf(" > population[%d].tours->total_distance: %f", p, initial_population[p].tours->total_distance);
		printf(" > population[%d].tours->node_qty: %d", p, initial_population[p].tours->node_qty);
		printf(" > population[%d].tours->nodes: %p", p, initial_population[p].tours->nodes);
		for (int t = 0; t < POPULATION_SIZE; ++t)
		{
			printf(" > population[%d].tours[%d]", p, t);
			printf(" > population[%d].tours[%d].fitness: %f", p, t, initial_population[p].tours[t].fitness);
			printf(" > population[%d].tours[%d].total_distance: %f", p, t, initial_population[p].tours[t].total_distance);
			printf(" > population[%d].tours[%d].node_qty: %d", p, t, initial_population[p].tours[t].node_qty);
			if (initial_population[p].tours[t].node_qty > 0)
			{
				printf(" > population[%d].tours[%d].nodes", p, t);
				printf(" > population[%d].tours[%d].nodes->id: %d", p, t, initial_population[p].tours[t].nodes->id);
				printf(" > population[%d].tours[%d].nodes->x: %f", p, t, initial_population[p].tours[t].nodes->x);
				printf(" > population[%d].tours[%d].nodes->y: %f", p, t, initial_population[p].tours[t].nodes->y);
				printf(" > population[%d].tours[%d].nodes->item_qty: %d", p, t, initial_population[p].tours[t].nodes->item_qty);
				printf(" > population[%d].tours[%d].nodes->items: %p", p, t, initial_population[p].tours[t].nodes->items);
				for (int n = 0; n < initial_population[p].tours[t].node_qty; ++n)
				{
					printf(" > population[%d].tours[%d].nodes[%d]:", p, t, n);
					printf(" > population[%d].tours[%d].nodes[%d].id: %d", p, t, n, initial_population[p].tours[t].nodes[n].id);
					printf(" > population[%d].tours[%d].nodes[%d].x: %f", p, t, n, initial_population[p].tours[t].nodes[n].x);
					printf(" > population[%d].tours[%d].nodes[%d].y: %f", p, t, n, initial_population[p].tours[t].nodes[n].y);
					printf(" > population[%d].tours[%d].nodes[%d].item_qty: %d", p, t, n, initial_population[p].tours[t].nodes[n].item_qty);
					if (initial_population[p].tours[t].nodes[n].item_qty > 0)
					{
						printf(" > population[%d].tours[%d].nodes[%d].items:", p, t, n);
						printf(" > population[%d].tours[%d].nodes[%d].items->id: %d", p, t, n, initial_population[p].tours[t].nodes[n].items->id);
						printf(" > population[%d].tours[%d].nodes[%d].items->node: %d", p, t, n, initial_population[p].tours[t].nodes[n].items->node);
						printf(" > population[%d].tours[%d].nodes[%d].items->taken: %d", p, t, n, initial_population[p].tours[t].nodes[n].items->taken);
						printf(" > population[%d].tours[%d].nodes[%d].items->value: %f", p, t, n, initial_population[p].tours[t].nodes[n].items->value);
						printf(" > population[%d].tours[%d].nodes[%d].items->weight: %f", p, t, n, initial_population[p].tours[t].nodes[n].items->weight);
						for (int i = 0; i < initial_population[p].tours[t].nodes[n].item_qty; ++i)
						{
							printf(" > population[%d].tours[%d].nodes[%d].items[%d]", p, t, n, i);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].id: %d", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].id);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].node: %d", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].node);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].taken: %d", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].taken);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].value: %f", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].value);
							printf(" > population[%d].tours[%d].nodes[%d].items[%d].weight: %f", p, t, n, i, initial_population[p].tours[t].nodes[n].items[i].weight);
						}
					}
				}
			}
		}
	}
	printf("\n\n");
}

/// <summary>
/// Function to test memory allocation in CUDA
/// </summary>
/// <param name="initial_population"></param>
/// <param name="population_size"></param>
/// <param name="node_quantity"></param>
/// <param name="item_quantity"></param>
/// <returns></returns>
void testMemoryAllocationCPU(population initial_population, const int population_size)
{
	printf(" > population.id: %d\n", initial_population.id);
	printf(" > population.tours: %p\n", initial_population.tours);
	printf(" > population.tours->fitness: %f\n", initial_population.tours->fitness);
	printf(" > population.tours->total_distance: %f\n", initial_population.tours->total_distance);
	printf(" > population.tours->node_qty: %d\n", initial_population.tours->node_qty);
	printf(" > population.tours->nodes: %p\n", initial_population.tours->nodes);
	for (int t = 0; t < POPULATION_SIZE; ++t)
	{
		printf(" > population.tours[%d]\n", t);
		printf(" > population.tours[%d].fitness: %f\n", t, initial_population.tours[t].fitness);
		printf(" > population.tours[%d].total_distance: %f\n", t, initial_population.tours[t].total_distance);
		printf(" > population.tours[%d].node_qty: %d\n", t, initial_population.tours[t].node_qty);
		if (initial_population.tours[t].node_qty > 0)
		{
			printf(" > population.tours[%d].nodes\n", t);
			printf(" > population.tours[%d].nodes->id: %d\n", t, initial_population.tours[t].nodes->id);
			printf(" > population.tours[%d].nodes->x: %f\n", t, initial_population.tours[t].nodes->x);
			printf(" > population.tours[%d].nodes->y: %f\n", t, initial_population.tours[t].nodes->y);
			printf(" > population.tours[%d].nodes->item_qty: %d\n", t, initial_population.tours[t].nodes->item_qty);
			printf(" > population.tours[%d].nodes->items: %p\n", t, initial_population.tours[t].nodes->items);
			for (int n = 0; n < initial_population.tours[t].node_qty; ++n)
			{
				printf(" > population.tours[%d].nodes[%d]\n", t, n);
				printf(" > population.tours[%d].nodes[%d].id: %d\n", t, n, initial_population.tours[t].nodes[n].id);
				printf(" > population.tours[%d].nodes[%d].x: %f\n", t, n, initial_population.tours[t].nodes[n].x);
				printf(" > population.tours[%d].nodes[%d].y: %f\n", t, n, initial_population.tours[t].nodes[n].y);
				printf(" > population.tours[%d].nodes[%d].item_qty: %d\n", t, n, initial_population.tours[t].nodes[n].item_qty);
				if (initial_population.tours[t].nodes[n].item_qty > 0)
				{
					printf(" > population.tours[%d].nodes[%d].items\n", t, n);
					printf(" > population.tours[%d].nodes[%d].items->id: %d\n", t, n, initial_population.tours[t].nodes[n].items->id);
					printf(" > population.tours[%d].nodes[%d].items->node: %d\n", t, n, initial_population.tours[t].nodes[n].items->node);
					printf(" > population.tours[%d].nodes[%d].items->taken: %d\n", t, n, initial_population.tours[t].nodes[n].items->taken);
					printf(" > population.tours[%d].nodes[%d].items->value: %f\n", t, n, initial_population.tours[t].nodes[n].items->value);
					printf(" > population.tours[%d].nodes[%d].items->weight: %f\n", t, n, initial_population.tours[t].nodes[n].items->weight);
					for (int i = 0; i < initial_population.tours[t].nodes[n].item_qty; ++i)
					{
						printf(" > population.tours[%d].nodes[%d].items[%d]\n", t, n, i);
						printf(" > population.tours[%d].nodes[%d].items[%d].id: %d\n", t, n, i, initial_population.tours[t].nodes[n].items[i].id);
						printf(" > population.tours[%d].nodes[%d].items[%d].node: %d\n", t, n, i, initial_population.tours[t].nodes[n].items[i].node);
						printf(" > population.tours[%d].nodes[%d].items[%d].taken: %d\n", t, n, i, initial_population.tours[t].nodes[n].items[i].taken);
						printf(" > population.tours[%d].nodes[%d].items[%d].value: %f\n", t, n, i, initial_population.tours[t].nodes[n].items[i].value);
						printf(" > population.tours[%d].nodes[%d].items[%d].weight: %f\n", t, n, i, initial_population.tours[t].nodes[n].items[i].weight);
					}
				}
			}
		}
	}
	printf("\n\n");
}