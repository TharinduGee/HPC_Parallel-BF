#ifndef UTILS_H
#define UTILS_H

typedef struct {
    int src;
    int dest;
    int weight;
} Edge;

Edge* generateEdges(int vertexCount, int edgeCount, int minWeight, int maxWeight);
void printEdges(Edge* edges, int edgeCount);

#endif