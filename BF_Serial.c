#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "time.h"
#include "utils/graphGeneratorUtils.h"

int bellmanFord(int n, Edge* edges, int edgeCount, int src, int* distance) {
     for (int i = 0; i < n; i++) {
          distance[i] = INT_MAX;
     }
     distance[src] = 0;
     int updated = 0;

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
                    updated = 1;
               }
          }
          if (!updated) break;
     }

     return 0;
}

int main() {
     int V, E, min_wt, max_wt;

     printf("Enter No of Verteces : ");
     scanf("%d", &V);
     printf("Enter minimum weight : ");
     scanf("%d", &min_wt);
     printf("Enter maximum weight : ");
     scanf("%d", &max_wt);

     generateGraph(V, min_wt, max_wt);
     Edge* edges = readGraphFromFile(V, min_wt, max_wt, &E);

     int src = 0;
     int* distance = (int*)malloc(V * sizeof(int));

     clock_t start = clock();
     int result = bellmanFord(V, edges, E, src, distance);
     clock_t end = clock();

     char filename[100];
     snprintf(filename, sizeof(filename), "serial_output__%d_%d_%d.txt", V, max_wt, min_wt);
     FILE *fp = fopen(filename, "w");
     if (!fp) {
          perror("Failed to write output file");
          return 1;
     }

     if (result == -1) {
          printf("Negative weight cycle detected.\n");
          fprintf(fp, "Negative weight cycle detected.\n");
     } else {
          for (int i = 1; i < V; i++) {
               fprintf(fp, "%d\n", distance[i]);
               if (distance[i] == INT_MAX) {
                    printf("No connection from source node - %d to destination node - %d \n", src, i);
               } else {
                    printf("Shortest distance from source node - %d to destination node - %d = %d\n", 
                         src, i, distance[i]);
               }
               
          }
     }

     double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
     printf("BF serial execution time : %f \n", time_spent);

     fclose(fp);
     printf("Output saved to file: %s\n", filename);

     free(edges);
     free(distance);
     return 0;
}