<!-- LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Sister20/if3230-tucil-siber">
    <img src="../../public/cuda-ic.svg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Tucil 3 - CUDA</h3>

  <p align="center">
    Matrix inverse solver using general-purpose computing on GPUs
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<br />
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
        <a href="#about-the-project">About The Project</a>
        <ul>
            <li><a href="#specification">Specification</a></li>
            <li><a href="#performance">Performance</a></li>
            <li><a href="#methods">Methods</a></li>
            <li><a href="#tech-stack">Tech Stack</a></li>
        </ul>
    </li>
    <li>
        <a href="#getting-started">Getting Started</a>
        <ul>
            <li><a href="#prerequisites">Prerequisites</a></li>
            <li><a href="#installation">Installation</a></li>
            <li><a href="#installation">Usage</a></li>
        </ul>
    </li>
    <li>
        <a href="#authors">Author</a>
    </li>
  </ol>
</details>
<br/>
<br/>

<!-- ABOUT THE PROJECT -->
## About The Project

Tucil 3 of Parallel and Distributed Systems (IF3230). Matrix inverse solver using ...

### Specification

* Program is able to read a square matrix from a text file
* Program is able to find the inverse matrix using gauss-jordan elimination
* Program is able to be ran in local machine using CUDA
* Program is able to write the inverse matrix into a text file
* Program has reasonable performance compared to its serial counterpart 

### Performance
Execution environment: Win11 - WSL2 Ubuntu 22.04 - 13th Gen Intel i9-13900H 2.60 GHz - 16GB Memory - NVIDIA GeForce RTX 4060

<table style="text-align: center;">
    <thead>
        <tr>
            <th scope="col" rowspan="2">Matrix</th>
            <th scope="col" colspan="2">Execution Time (s)</th>
            <th scope="col" rowspan="2">Speed Up</th>
        </tr>
        <tr>
            <th scope="col">Serial</th>
            <th scope="col">CUDA</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>32</td>
            <td>0.02</td>
            <td>0.25</td>
            <td>0.08</td>
        </tr>
        <tr>
            <td>64</td>
            <td>0.04</td>
            <td>0.45</td>
            <td>0.09</td>
        </tr>
        <tr>
            <td>128</td>
            <td>0.12</td>
            <td>0.39</td>
            <td>0.31</td>
        </tr>
        <tr>
            <td>256</td>
            <td>0.51</td>
            <td>0.50</td>
            <td>1.02</td>
        </tr>
        <tr>
            <td>512</td>
            <td>2.54</td>
            <td>1.74</td>
            <td>1.46</td>
        </tr>
        <tr>
            <td>1024</td>
            <td>13.65</td>
            <td>6.25</td>
            <td>2.18</td>
        </tr>
        <tr>
            <td>2048</td>
            <td>86.15</td>
            <td>24.22</td>
            <td>3.56</td>
        </tr>
    </tbody>
</table>

Execution environment: Win11 - WSL2 Ubuntu 22.04 - AMD Ryzen 5 5600H Radeon Graphics - 16GB Memory - NVIDIA GeForce RTX 3050

<table style="text-align: center;">
    <thead>
        <tr>
            <th scope="col" rowspan="2">Matrix</th>
            <th scope="col" colspan="2">Execution Time (s)</th>
            <th scope="col" rowspan="2">Speed Up</th>
        </tr>
        <tr>
            <th scope="col">Serial</th>
            <th scope="col">CUDA</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>2048</td>
            <td>86.15</td>
            <td>9.086</td>
            <td>9.482</td>
        </tr>
    </tbody>
</table>

### Methods

1. Determine block size based on the dimension of the matrix for optimal GPU performance

2. Read the matrix from a text file and store it in the CPU

3. Initialize the identity matrix in the CPU

4. Allocate memory to store the initial matrix, the result, and the identity matrix in the GPU

5. Perform Gauss-Jordan Elimination in the GPU using parallelization:

    - normalizeTransform(): Normalize each row of the matrix by dividing every non-pivot element with the pivot
    - transformToUnit(): Divide the pivot of each row to get every pivot of the each row equals to 1
    - transformToDiagonal(): Eliminate non-pivot element to form a row-echelon matrix (identity matrix)

6. The chosen approach iterates through each row and performs three calls: normalizeTransform, transformToUnit, and transformToDiagonal. These operations depend on the pivot values in the current row. Processing elements within a row leverages the shared memory effectively. Threads within a block can efficiently access and share the pivot element related to the current row stored in shared memory.

7. Move the result matrix into the CPU from the GPU

8. Free the memory allocated in the GPU and in the CPU

### Built With

* [![Cpp][Cpp.cpp]][Cpp-url]
* [CUDA][CUDA-url]

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites
* Linux operating system or WSL
* G++ for C++ compilation
* Make to run makefile
* Open MPI library

### Installation
1. Clone the repo
  <br/>

   ```sh
   git clone https://github.com/Sister20/if3230-tucil-siber/tree/main
   ```
2. Go to the repository root folder `if3230-tucil-siber`
  <br/>

   ```sh
   cd if3230-tucil-siber
   ```
3. Run the program `cuda.cu` using makefile
  <br/>

   ```sh
   make cuda
   ```
<br/>
<br/>

### Usage

1. Create a text file containing the matrix in the `test_cases` folder with the number of rows as the file name
    <br/>

    ```ssh
    /* 8.txt */

    8
    7 6 6 3 9 5 7 2
    3 1 6 5 1 4 6 4
    0 6 5 6 9 2 3 8
    6 3 9 0 1 1 5 1
    6 7 5 6 1 6 1 9
    6 8 6 1 4 8 7 8
    2 1 0 9 0 2 9 1
    9 8 3 4 7 1 9 1
    ```
2. Define the size in the `makefile`
    <br/>

    ```ssh
    ...
    SIZE = 8
    ...
    ```
3. Run the program
    <br/> 

    ```ssh
    make cuda
    ```
4. The result is stored inside `result` folder as `cuda_8.txt`
    <br/>

    ```ssh
    /* cuda_8.txt */

    -0.101528 0.459338 -0.114191 -0.22553 0.0313288 -0.0982821 -0.286578 0.295637
    0.121358 -0.702845 0.0804064 0.322938 0.0787497 0.101702 0.354614 -0.274504
    0.083176 -0.183177 0.0490253 0.207183 0.0197436 -0.0186246 0.103854 -0.16558
    0.0886022 -0.132632 0.0218045 0.0587376 0.0891459 -0.0902586 0.132458 -0.092554
    0.0390927 0.198307 0.0202419 -0.139824 -0.0483205 -0.0515535 -0.140788 0.0945772
    0.218977 -0.20277 -0.0567472 0.0309494 0.0458075 0.0584846 0.140008 -0.224
    -0.0980289 0.0944584 0.00650595 -0.0227517 -0.108984 0.0862361 0.00617166 0.0737198
    -0.271418 0.533268 0.006676 -0.25765 -0.054484 0.0140958 -0.309142 0.300737
    ```
<br/>
<br/>

<!-- AUTHOR -->

## Authors

| NIM | Name | 
| :---: | :---: |
| 13521019 | Ditra Rizqa Amadia | 
| 13521021 | Bernardus Willson |
| 13521031 | Fahrian Afdholi |

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[Cpp-url]: https://isocpp.org/std/the-standard
[Cpp.cpp]: https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white
[CUDA-url]: https://developer.nvidia.com/cuda-zone