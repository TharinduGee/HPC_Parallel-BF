#include <stdio.h>
#include <limits.h>

typedef struct Edge {
     int u, v, wt
} Edge;

int bellmanFord(int n, Edge* edges, int edgeCount, int src, int* distance) {
     for (int i = 0; i < n; i++) {
          distance[i] = INT_MAX;
     }
     distance[src] = 0;

     for (int i = 0; i < n; i++) {
          for (int j = 0; j < edgeCount; j++) {
               int u = edges[j].u;
               int v = edges[j].v;
               int wt = edges[j].wt;
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
    int n = 5; 
    int edgeCount = 5; 

    Edge* edges = (Edge*) malloc(edgeCount * sizeof(Edge));

    edges[0] = (Edge){1, 3, 2};
    edges[1] = (Edge){4, 3, -1};
    edges[2] = (Edge){2, 4, 1};
    edges[3] = (Edge){1, 2, 1};
    edges[4] = (Edge){0, 1, 5};

    int src = 0;
    int* distance = (int*)malloc(n * sizeof(int));

    int result = bellmanFord(n, edges, edgeCount, src, distance);

    if (result == -1) {
        printf("Negative weight cycle detected.\n");
    } else {
        for (int i = 1; i < n; i++) {
            printf("Shortest distance from source node - %d to destination node - %d = %d\n",
                   src, i, distance[i]);
        }
    }

    free(edges);
    free(distance);
    return 0;
}