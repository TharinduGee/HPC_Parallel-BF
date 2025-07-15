#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "graphGeneratorUtils.h"
 
#define SEED 30
 
typedef unsigned char vertex;

void generateGraph(int nV, int minWeight, int maxWeight) {
     if (nV <= 0) return;

     char fileName[100];
     snprintf(fileName, sizeof(fileName), "graph_%d_%d_%d.txt", nV, maxWeight, minWeight);

     FILE* fileCheck = fopen(fileName, "r");
     if (fileCheck != NULL) {
          fclose(fileCheck);
          printf("File '%s' already exists.\n", fileName);
          return;
     }

     int maxNumberOfEdges = nV/2;
     srand(SEED);

     vertex ***graph;
     int **weights;
     int totEdges = 0;
 
     if ((graph = (vertex ***) malloc(sizeof(vertex **) * nV)) == NULL){
          printf("Memory allocation for graph is failed\n");
          exit(1);
     }
     if ((weights = (int **) malloc(sizeof(int *) * nV)) == NULL){
          printf("Memory allocation for weight matrix is failed\n");
     }
 
     for (int v = 0; v < nV; v++){
          if ((graph[v] = (vertex **) malloc(sizeof(vertex *) * maxNumberOfEdges)) == NULL) {
               printf("Memory allocation for vertices is failed\n");
               exit(1);
          }
          if ((weights[v] = (int *) malloc(sizeof(int) * maxNumberOfEdges)) == NULL) {
               printf("Memory allocation for weights is failed\n");
               exit(1);
          }
          
          for (int e = 0; e < maxNumberOfEdges; e++) {
               if ((graph[v][e] = (vertex *) malloc(sizeof(vertex))) == NULL) {
                    printf("Memory allocation for edges is failed\n");
                    exit(1);
               }
          }
     }

     FILE* file = fopen(fileName, "w");
     if (file == NULL) {
          printf("Failed to open file '%s' for writing\n", fileName);
          exit(1);
     }
     fprintf(file, "%d\n", nV);
     
     for (int v = 0; v < nV; v++){
          fprintf(file, "%d:", v);
          int *connected = calloc(nV, sizeof(int));

          for (int e = 0; e < maxNumberOfEdges; e++){
               if (rand() % 2 == 1){
                    int linkedVertex;

                    while ((linkedVertex = rand() % nV) == v || connected[linkedVertex]);
                    connected[linkedVertex] = 1;
                    graph[v][e] = *graph[linkedVertex];
                    weights[v][e] = rand() % (minWeight + maxWeight + 1) + minWeight;
                    
                    fprintf(file, "%d,%d;", linkedVertex, weights[v][e]);
                    totEdges++;
               }
               else{ 
                    graph[v][e] = NULL;
                    weights[v][e] = 0;
               }
          }
          free(connected);
          fprintf(file, "\n");
     }

     fclose(file);
     printf("Graph file '%s' has been generated with %d vertices and %d total edges.\n", fileName, nV, totEdges);
}

Edge* readGraphFromFile(int v, int minWeight, int maxWeight, int* edgeCountOut) {
     if (v <= 0) return NULL;

     char fileName[100];
     snprintf(fileName, sizeof(fileName), "graph_%d_%d_%d.txt", v, maxWeight, minWeight);
     printf("Using the existing file...\n");

     FILE* file = fopen(fileName, "r");
     if (!file) {
          printf("Error: Could not open file '%s'\n", fileName);
          return NULL;
     }

     int nV;
     fscanf(file, "%d\n", &nV);

     int edgeCapacity = nV * (nV - 1) / 2;
     Edge* edges = (Edge*)malloc(sizeof(Edge) * edgeCapacity);
     if (!edges) {
          printf("Memory allocation failed for edges\n");
          fclose(file);
          return NULL;
     }

     int edgeCount = 0;
     char line[1024];
     while (fgets(line, sizeof(line), file)) {
          char* colon = strchr(line, ':');
          if (!colon) continue;

          *colon = '\0';  
          int src = atoi(line);
          char* edgeList = colon + 1;

          char* token = strtok(edgeList, ";\n");
          while (token != NULL) {
               int dest, weight;
               if (sscanf(token, "%d,%d", &dest, &weight) == 2) {
                    edges[edgeCount].src = src;
                    edges[edgeCount].dest = dest;
                    edges[edgeCount].weight = weight;
                    edgeCount++;
               }
               token = strtok(NULL, ";\n");
          }
     }

     fclose(file);
     *edgeCountOut = edgeCount; 
     return edges;
}