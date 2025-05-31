#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <limits.h>
#include "utils/utils.h"

int bellmanFord(int n, Edge* edges, int edgeCount, int src, int* distance) {

     #pragma omp parallel for
     for (int i=0; i < n; i++) {
          distance[i] = INT_MAX;
     }
     distance[src] = 0;

     for (int i = 0; i < n; i++) {
          int updated = 0;

          #pragma omp parallel for shared(distance, updated)
          for (int j = 0; j < edgeCount; j++) {
               int u = edges[j].src;
               int v = edges[j].dest;
               int wt = edges[j].weight;
               #pragma omp critical
               {
                    if (distance[u] != INT_MAX && distance[u] + wt < distance[v]) {
                         distance[v] = distance[u] + wt;
                         updated = 1;
                    }
               }
          }

          if (!updated) break;

          if (i == n - 1 && updated) {
               return -1;
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

     double start = omp_get_wtime();
     int result = bellmanFord(V, edges, E, src, distance);
     double end = omp_get_wtime();

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

     double time_spent = (double)(end - start);
     printf("BF parallel openmp execution time : %f \n", time_spent);

     free(edges);
     free(distance);
     return 0;
}