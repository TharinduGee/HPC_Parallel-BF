#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include "utils/graphGeneratorUtils.h"

#define BLOCK_DIM 256

// devide edges to cpu and gpu according to partition point
void partitionEdges(Edge* global_edges, int edgeCount, int partition_point, 
     Edge** cpu_edges_out, int* cpu_edge_count_out, 
     Edge** gpu_edges_out, int* gpu_edge_count_out) {
    
     int cpu_count = 0;
     int gpu_count = 0;
     for (int i = 0; i < edgeCount; i++) {
          if (global_edges[i].dest < partition_point) {
               cpu_count++;
          } else {
               gpu_count++;
          }
     }

     *cpu_edge_count_out = cpu_count;
     *gpu_edge_count_out = gpu_count;
     *cpu_edges_out = (Edge*) malloc(cpu_count * sizeof(Edge));
     *gpu_edges_out = (Edge*) malloc(gpu_count * sizeof(Edge));

     int cpu_idx = 0;
     int gpu_idx = 0;
     for (int i = 0; i < edgeCount; i++) {
          if (global_edges[i].dest < partition_point) {
               (*cpu_edges_out)[cpu_idx++] = global_edges[i];
          } else {
               (*gpu_edges_out)[gpu_idx++] = global_edges[i];
          }
     }
     printf("Partitioning complete: CPU gets %d edges, GPU gets %d edges.\n", 
          cpu_count, gpu_count);
}

// using same edge relaxaion kernel from cuda parallel version
__global__ void relax_gpu_kernel(Edge *d_edges, int *d_distance, int edgeCount, int* d_updated) {
     int j = blockIdx.x * blockDim.x + threadIdx.x;
     if (j < edgeCount) {
          int u = d_edges[j].src;
          int v = d_edges[j].dest;
          int wt = d_edges[j].weight;
        
          if (d_distance[u] != INT_MAX) {
               int new_dist = d_distance[u] + wt;
               int old_dist = atomicMin(&d_distance[v], new_dist);
               if (new_dist < old_dist) {
                    *d_updated = 1;
               }
          }
     }
}

// using similar edge relaxation logic from openmp parallel version
void relax_cpu_partition(Edge* h_edges, int edgeCount, int* h_distance, int* h_updated) {
     
     #pragma omp parallel for
     for (int j = 0; j < edgeCount; j++) {
          int u = h_edges[j].src;
          int v = h_edges[j].dest;
          int wt = h_edges[j].weight;
        
          if (h_distance[u] != INT_MAX) {
               int new_dist = h_distance[u] + wt;
            
               #pragma omp critical
               {
                    if (new_dist < h_distance[v]) {
                         h_distance[v] = new_dist;
                         *h_updated = 1;
                    }
               }
          }
     }
}

int bellmanFord(int n, int k, Edge* cpu_edges, int cpu_edge_count, 
     Edge* gpu_edges, int gpu_edge_count, int src, int* final_distance) {
    
     int* h_distance = (int*)malloc(n * sizeof(int));

     Edge* d_gpu_edges;
     int *d_distance, *d_updated;
     cudaMalloc(&d_gpu_edges, gpu_edge_count * sizeof(Edge));
     cudaMalloc(&d_distance, n * sizeof(int));
     cudaMalloc(&d_updated, sizeof(int));

     for(int i = 0; i < n; i++) {
          h_distance[i] = (i == src) ? 0 : INT_MAX;
     }

     cudaMemcpy(d_distance, h_distance, n * sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_gpu_edges, gpu_edges, gpu_edge_count * sizeof(Edge), cudaMemcpyHostToDevice);
    
     cudaStream_t stream;
     cudaStreamCreate(&stream);

     for (int i = 0; i < n; i++) {
          int h_updated_val = 0;
          int d_updated_val = 0;

          cudaMemcpyAsync(d_updated, &d_updated_val, sizeof(int), cudaMemcpyHostToDevice, stream);
        
          relax_gpu_kernel<<<(gpu_edge_count + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM, 0, stream>>>(
               d_gpu_edges, d_distance, gpu_edge_count, d_updated);

          relax_cpu_partition(cpu_edges, cpu_edge_count, h_distance, &h_updated_val);

          cudaStreamSynchronize(stream);

          cudaMemcpy(h_distance + k, d_distance + k, (n - k) * sizeof(int), cudaMemcpyDeviceToHost);
        
          cudaMemcpy(d_distance, h_distance, n * sizeof(int), cudaMemcpyHostToDevice);

          cudaMemcpy(&d_updated_val, d_updated, sizeof(int), cudaMemcpyDeviceToHost);
          int overall_updated = h_updated_val || d_updated_val;

          if (i == n - 1 && overall_updated) {
               free(h_distance);
               cudaFree(d_gpu_edges); 
               cudaFree(d_distance); 
               cudaFree(d_updated);
               cudaStreamDestroy(stream);
               return -1;
          }
        
          if (!overall_updated) {
               break;
          }
     }

     cudaMemcpy(final_distance, d_distance, n * sizeof(int), cudaMemcpyDeviceToHost);
    
     free(h_distance);
     cudaFree(d_gpu_edges);
     cudaFree(d_distance);
     cudaFree(d_updated);
     cudaStreamDestroy(stream);

     return 0;
}


int main() {

     int V, E, min_wt, max_wt;
     printf("Enter No of Vertices : "); 
     scanf("%d", &V);
     printf("Enter minimum weight : "); 
     scanf("%d", &min_wt);
     printf("Enter maximum weight : "); 
     scanf("%d", &max_wt);

     generateGraph(V, min_wt, max_wt);
     Edge* global_edges = readGraphFromFile(V, min_wt, max_wt, &E);

     int src = 0; 
     float cpu_gpu_ratio = 0.00;
     int* distance = (int*) malloc(V * sizeof(int));

     printf("Enter cpu to cpu ratio to share edge relaxation : ");
     scanf("%f", &cpu_gpu_ratio);

     Edge *cpu_edges, *gpu_edges;
     int cpu_edge_count, gpu_edge_count;
     int partition = cpu_gpu_ratio * V;

     double start_time = omp_get_wtime();

     partitionEdges(global_edges, E, partition, &cpu_edges, &cpu_edge_count, 
          &gpu_edges, &gpu_edge_count);
     int result = bellmanFord(V, partition, cpu_edges, cpu_edge_count, 
          gpu_edges, gpu_edge_count, src, distance);

     double end_time = omp_get_wtime();

     char filename[100];
     snprintf(filename, sizeof(filename), "hybrid_output__%d_%d_%d.txt", V, max_wt, min_wt);
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
    
     printf("\nHybrid execution time: %f seconds\n", end_time - start_time);

     fclose(fp);
     printf("Output saved to file: %s\n", filename);

     free(global_edges);
     free(distance);
     free(cpu_edges);
     free(gpu_edges);

     return 0;
}