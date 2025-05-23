#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include "utils.h"

#define SEED 42

Edge* generateEdges(int vertexCount, int edgeCount, int minWeight, int maxWeight) {
    if (vertexCount <= 0 || edgeCount <= 0 || minWeight >= maxWeight) {
        fprintf(stderr, "Invalid input parameters\n");
        return NULL;
    }

    srand(SEED);

    Edge* edges = (Edge*)malloc(edgeCount * sizeof(Edge));
    if (edges == NULL) {
        fprintf(stderr, "Memory allocation failed for edges\n");
        return NULL;
    }

    bool** connections = (bool**)malloc(vertexCount * sizeof(bool*));
    for (int i = 0; i < vertexCount; i++) {
        connections[i] = (bool*)calloc(vertexCount, sizeof(bool));
    }

    int generatedEdges = 0;
    while (generatedEdges < edgeCount) {
        int src = rand() % vertexCount;
        int dest = rand() % vertexCount;

        if (src == dest || connections[src][dest] || connections[dest][src]) {
            continue;
        }

        edges[generatedEdges].src = src;
        edges[generatedEdges].dest = dest;
        edges[generatedEdges].weight = minWeight + (rand() % (maxWeight - minWeight + 1));

        connections[src][dest] = true;
        connections[dest][src] = true;

        generatedEdges++;
    }

    for (int i = 0; i < vertexCount; i++) {
        free(connections[i]);
    }
    free(connections);

    return edges;
}

void printEdges(Edge* edges, int edgeCount) {
    printf("\nGraph Edges (Adjacency List Format):\n");
    for (int i = 0; i < edgeCount; i++) {
        printf("%d -> %d (weight %d)\n", edges[i].src, edges[i].dest, edges[i].weight);
    }
}