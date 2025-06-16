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

Edge* generateEdges(int vertexCount, int edgeCount, int minWeight, int maxWeight);
void printEdges(Edge* edges, int edgeCount);

#ifdef __cplusplus
}
#endif

#endif