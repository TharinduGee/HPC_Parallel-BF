#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "utils/utils.h"

int bellmanFord(int n, Edge* edges, int edgeCount, int src, int* distance) {
     for (int i = 0; i < n; i++) {
          distance[i] = INT_MAX;
     }
     distance[src] = 0;

     for (int i = 0; i < n; i++) {
          for (int j = 0; j < edgeCount; j++) {
               int u = edges[j].src;
               int v = edges[j].dest;
               int wt = edges[j].weight;
               if (distance[u] != INT_MAX && distance[u] + wt < distance[v]) {
                    if (i == n - 1) {
                         return -1;
                    }
                    distance[v] = distance[u] + wt;
               }
          }
     }

     return 0;
}

int main() {
     int V, E, min_wt, max_wt;

     printf("Enter No of Verteces : ");
     scanf("%d", &V);
     printf("Enter No of Edges : ");
     scanf("%d", &E);
     printf("Enter minimum weight : ");
     scanf("%d", &min_wt);
     printf("Enter maximum weight : ");
     scanf("%d", &max_wt);

     Edge* edges = generateEdges(V, E, min_wt, max_wt);

     int src = 0;
     int* distance = (int*)malloc(V * sizeof(int));

     int result = bellmanFord(V, edges, E, src, distance);

     if (result == -1) {
          printf("Negative weight cycle detected.\n");
     } else {
          for (int i = 1; i < V; i++) {
               if (distance[i] == INT_MAX) {
                    printf("No connection from source node - %d to destination node - %d \n", src, i);
               } else {
                    printf("Shortest distance from source node - %d to destination node - %d = %d\n", 
                         src, i, distance[i]);
               }
               
          }
     }

     free(edges);
     free(distance);
     return 0;
}