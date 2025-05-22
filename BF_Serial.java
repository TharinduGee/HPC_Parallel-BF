import java.util.Arrays;

public class BF_Serial {

     static int[] bellmanFord(int n, int[][] edges, int src) {
          // Initilize distance to all other vectors to Infinete except source
          int[] distance = new int[n];
          Arrays.fill(distance, Integer.MAX_VALUE);
          distance[src] = 0;
          
          // Edge relaxation
          for(int i=0; i<n; i++) {
               for (int[] edge : edges) {
                    int u = edge[0];
                    int v = edge[1];
                    int wt = edge[2];
                    if (distance[u] != Integer.MAX_VALUE && distance[u] + wt < distance[v]) {
                         // Detect negetive cycle
                         if (i == n-1) 
                              return new int[]{-1};

                         distance[v] = distance[u] + wt; 
                    }

               }
          }


          return distance;
     }
     
     public static void main(String[] args) {

          int n = 5;
          int[][] edges = new int[][] {
               {1, 3, 2},    
               {4, 3, -1},   
               {2, 4, 1},    
               {1, 2, 1},    
               {0, 1, 5}     
          };
          int src = 0;

          int[] ans = bellmanFord(n, edges, src);

          for (int i=1; i<n; i++) {
               System.out.printf("Shortest distance from source node - %d to destination node - %d = %d \n", src, i, ans[i]);
          }
     }

}

