#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "utils/graphGeneratorUtils.h"

#define BLOCK_DIM 256

__global__ void initialize(int n, int src, int* d_distance) {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < n) {
          d_distance[i] = (i == src) ? 0 : INT_MAX;
     }
}

__global__ void relax(Edge *edges, int *d_distance, int edgeCount, int* d_updated) {
     int j = blockIdx.x * blockDim.x + threadIdx.x;
     if (j < edgeCount) {
          int u = edges[j].src;
          int v = edges[j].dest;
          int wt = edges[j].weight;
          // check whether the vertice is reacheable
          if (d_distance[u] != INT_MAX) {
               int new_dist = d_distance[u] + wt;
               // atomically update minimum across all threads but returns old distance
               int curr_dist = atomicMin(&d_distance[v], new_dist);
               if (new_dist < curr_dist) {
                    *d_updated = 1; 
               }
          }
     }
}

int bellmanFord(int n, Edge* edges, int edgeCount, int src, int* distance) {

     int *d_distance;
     cudaMalloc(&d_distance, n * sizeof(int));
     cudaMemcpy(d_distance, distance, n * sizeof(int), cudaMemcpyHostToDevice);
     initialize<<<(n - 1 + BLOCK_DIM)/BLOCK_DIM, BLOCK_DIM>>>(n, src, d_distance);
     // wait till initialization is completed
     cudaDeviceSynchronize();

     int updated;
     int* d_updated;
     Edge* d_edges;
     cudaMalloc(&d_updated, sizeof(int));
     cudaMalloc(&d_edges, edgeCount * sizeof(Edge));
     cudaMemcpy(d_edges, edges, edgeCount * sizeof(Edge), cudaMemcpyHostToDevice);

     for (int i = 0; i < n; i++) {
          updated = 0;
          cudaMemcpy(d_updated, &updated, sizeof(int), cudaMemcpyHostToDevice);
          relax<<<(edgeCount - 1 + BLOCK_DIM)/BLOCK_DIM, BLOCK_DIM>>>(d_edges, d_distance, edgeCount, d_updated);
          cudaMemcpy(&updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost);

          // detect negetive cycle
          if (i == n - 1 && updated) {
               cudaFree(d_edges);
               cudaFree(d_distance);
               cudaFree(d_updated);
               return -1;
          }
          
          // early stopping
          if (!updated) break;
     }

     cudaMemcpy(distance, d_distance, n * sizeof(int), cudaMemcpyDeviceToHost);

     cudaFree(d_edges);
     cudaFree(d_distance);
     cudaFree(d_updated);

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

     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);

     cudaEventRecord(start);
     int result = bellmanFord(V, edges, E, src, distance);
     cudaEventRecord(stop);

     cudaEventSynchronize(stop);
     float time_spent = 0;
     cudaEventElapsedTime(&time_spent, start, stop);

     char filename[100];
     snprintf(filename, sizeof(filename), "cuda_output__%d_%d_%d.txt", V, max_wt, min_wt);
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

     printf("BF parallel cuda execution time : %f \n", time_spent / 1000.0f);

     fclose(fp);
     printf("Output saved to file: %s\n", filename);
     
     free(edges);
     free(distance);
     return 0;
}