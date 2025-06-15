#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "time.h"
#include "utils/utils.h"

#define BLOCK_DIM 256

__global__ initialize(int n, int src, int* d_distance) {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < n) {
          d_distance[i] = (i == src) ? 0 : INT_MAX;
     }
}

__global__ relax(Edge *edges, int *d_distance, int edgeCount, int* d_updated) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < edgeCount) {
        int u = edges[j].src;
        int v = edges[j].dest;
        int wt = edges[j].weight;
        if (d_distance[u] != INT_MAX && d_distance[v] > d_distance[u] + wt) {
            d_distance[v] = d_distance[u] + wt;
            d_updated = 1;
        }
    }
}

int bellmanFord(int n, Edge* edges, int edgeCount, int src, int* distance) {

     int *d_distance;
     cudaMalloc(&d_distance, n * sizeof(int))
     cudaMemcpy(d_distance, distance, n * sizeof(int), cudaMemcpyHostToDevice);
     initialize<<<(n - 1 + BLOCK_DIM)/BLOCK_DIM, BLOCK_DIM>>>(n, src, d_distance);
     // wait till initialization is completed
     cudaDeviceSynchronize();

     int updated;
     int* d_updated;
     Edge* d_edges;
     cudaMalloc(&d_updated, sizeof(int));
     cudaMalloc(&d_edged, edgecount * sizeof(Edge));
     cudaMemcpy(d_edges, edged, edgeCount * sizeof(Edge), cudaMemcpyHostToDevice);

     for (int i = 0; i < n; i++) {
          updated = 0;
          cudaMemcpy(d_updated, updated, sizeof(int), cudaMemcpyHostToDevice);
          relax<<<edgeCount - 1 + BLOCK_DIM, BLOCK_DIM>>>(d_edges, d_distance, edgeCount, d_updated);
          cudaMemcpy(updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost);
          cudaDeviceSynchronize();

          // detect negetive cycle
          if (i == n - 1 && updated) {
               cudaFree(d_edges);
               cudaFree(d_distance);
               cudaFree(d_updated);
               return -1;
          }
          
          // early stopping
          if (!d_updated) break;
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
     printf("Enter No of Edges : ");
     scanf("%d", &E);
     printf("Enter minimum weight : ");
     scanf("%d", &min_wt);
     printf("Enter maximum weight : ");
     scanf("%d", &max_wt);

     Edge* edges = generateEdges(V, E, min_wt, max_wt);

     int src = 0;
     int* distance = (int*)malloc(V * sizeof(int));

     clock_t start = clock();
     int result = bellmanFord(V, edges, E, src, distance);
     clock_t end = clock();

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

     double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
     printf("BF serial execution time : %f \n", time_spent);

     free(edges);
     free(distance);
     return 0;
}