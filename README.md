# **🧮 High Performance Computing for Bellman-Ford Algorithm**

This repository presents multiple implementations of the Bellman-Ford algorithm designed for the High Performance Computing (HPC) module. The objective is to explore and evaluate different parallel programming paradigms—serial, OpenMP, CUDA, and hybrid models—to solve the single-source shortest path problem on graphs with negative weights.

## **📂 Project Overview**

The project includes the following versions:

| Version | Description | Technology Used |
| :---- | :---- | :---- |
| BF\_Serial.c | Sequential implementation | C |
| BF\_Parallel\_openmp.c | Shared-memory parallelism using OpenMP | OpenMP |
| BF\_Parallel\_cuda.cu | GPU-accelerated Bellman-Ford using CUDA | CUDA |
| BF\_Hybrid\_Parallel.cu | Hybrid CPU-GPU approach using OpenMP and CUDA | OpenMP \+ CUDA |

Each version is benchmarked to evaluate improvements in execution time and scalability.

## **📂 Directory Structure**

.  
├── BF\_Serial.c  
├── BF\_Parallel\_openmp.c  
├── BF\_Parallel\_cuda.cu  
├── BF\_Hybrid\_Parallel.cu  
├── utils/  
│   ├── graphGeneratorUtils.c  
│   ├── graphGeneratorUtils.h  
│   └── (Other helper and utility functions)  
└── README.md

## **⚙️ Compilation and Execution Instructions**

### **📌 Prerequisites**

Ensure you have installed:

* GCC with OpenMP support  
* NVIDIA CUDA Toolkit (nvcc)  
* A CUDA-capable GPU (for CUDA and hybrid versions)

### **🔹 Serial Version (Baseline)**

gcc BF\_Serial.c utils/\*.c \-o BF\_Serial.exe  
./BF\_Serial.exe

### **🔹 OpenMP Parallel Version (Multi-core CPU)**

gcc \-fopenmp BF\_Parallel\_openmp.c utils/\*.c \-o BF\_Parallel\_openmp.exe  
OMP\_NUM\_THREADS=4 ./BF\_Parallel\_openmp.exe

**Note:** Set OMP\_NUM\_THREADS to control the thread count.

### **🔹 CUDA Version (GPU Acceleration)**

nvcc BF\_Parallel\_cuda.cu utils/\*.c \-I./utils \-o BF\_Parallel\_cuda.exe  
./BF\_Parallel\_cuda.exe

### **🔹 Hybrid Version (CPU-GPU Collaboration)**

nvcc BF\_Hybrid\_Parallel.cu utils/\*.c \-I./utils \-Xcompiler \-fopenmp \-o BF\_Hybrid\_Parallel.exe  
OMP_NUM_THREADS=4 ./BF\_Hybrid\_Parallel.exe

## **🧪 Graph Generation and Input**

To ensure fair and consistent benchmarking, the program uses an automated graph generation and loading system.

### **How It Works**

1. **Check for Existing Graph:** When you run any of the executables, the program first checks if a graph file with a specific name already exists in the directory.  
2. **Generate if Needed:** If the file does not exist, it calls the functions located in utils/graphGeneratorUtils.c to create a new random graph.  
3. **Load Graph:** The program then loads the graph data from this file into memory for the Bellman-Ford algorithm to process.

### **Naming Convention**

The generated graph files follow a strict naming pattern to avoid regeneration and ensure consistency:

graph\_\<VertexCount\>\_\<MaxWeight\>\_\<MinWeight\>.txt

**Example:**

If you set num\_vertices \= 100, max\_weight \= 30, and min\_weight \= \-30, the program will look for or create a file named:

graph\_100\_30\_-30.txt

This reusability is a key feature, as it guarantees that all four versions (Serial, OpenMP, CUDA, Hybrid) are tested on the exact same input graph, making performance comparisons valid.

## **📊 Features**

* 🔁 **Reusable Graph Files:** Avoids unnecessary regeneration for consistent benchmarks.  
* 📦 **Modular Design:** Common graph logic is separated into the utils/ directory.  
* ⚙️ **Dynamic Graph Generation:** Graph properties (vertices, weights) are easily configured in the runtime.  
* 📈 **Benchmark-Ready:** Designed for easy comparison of runtimes between different programming models.