#ifndef UTILS_H
#define UTILS_H

typedef struct {
    int src;
    int dest;
    int weight;
} Edge;

#ifdef __cplusplus
extern "C" {
#endif

void generateGraph(int nV, int minWeight, int maxWeight);
Edge* readGraphFromFile(int nV, int minWeight, int maxWeight, int* edgeCount);

#ifdef __cplusplus
}
#endif

#endif