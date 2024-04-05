<!-- LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Sister20/if3230-tucil-siber">
    <img src="../../public/open-mpi-ic.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Tucil 1 - Open MPI</h3>

  <p align="center">
    Matrix inverse solver using distributed memory and message passing.
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

Tucil 1 of Parallel and Distributed Systems (IF3230). Matrix inverse solver using distributed memory and message passing. 

### Specification

* Program is able to read a square matrix from a text file
* Program is able to find the inverse matrix using gauss-jordan elimination
* Program is able to be ran in the server using 4 nodes
* Program is able to write the inverse matrix into a text file
* Program has reasonable performance compared to its serial counterpart 

### Performance
Execution environment: Win11 - WSL2 Ubuntu 22.04 - 13th Gen Intel i9-13900H 2.60 GHz - 16GB Memory 

<table style="text-align: center;">
    <thead>
        <tr>
            <th scope="col" rowspan="2">Matrix</th>
            <th scope="col" colspan="2">Execution Time (s)</th>
            <th scope="col" rowspan="2">Speed Up</th>
        </tr>
        <tr>
            <th scope="col">Serial</th>
            <th scope="col">Open MPI</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>32</td>
            <td>0.02</td>
            <td>0.31</td>
            <td>0.07</td>
        </tr>
        <tr>
            <td>64</td>
            <td>0.04</td>
            <td>0.32</td>
            <td>0.13</td>
        </tr>
        <tr>
            <td>128</td>
            <td>0.12</td>
            <td>0.33</td>
            <td>0.36</td>
        </tr>
        <tr>
            <td>256</td>
            <td>0.51</td>
            <td>0.4</td>
            <td>1.28</td>
        </tr>
        <tr>
            <td>512</td>
            <td>2.54</td>
            <td>0.79</td>
            <td>3.22</td>
        </tr>
        <tr>
            <td>1024</td>
            <td>13.65</td>
            <td>2.72</td>
            <td>5.02</td>
        </tr>
        <tr>
            <td>2048</td>
            <td>86.15</td>
            <td>16.85</td>
            <td>5.11</td>
        </tr>
    </tbody>
</table>

### Methods

1. The program initializes the MPI library, enabling parallelization, that is multiple processes running on different nodes to communicate and coordinate their works
    
    Open MPI library initializes the mpi world, then runs every defined processes labeled by a unique rank. The program are partitioned, thus each process works on a portion of the overall program. These partitioning are defined explicitly inside the program. Each process performs its computation on its own local data as the memory is distributed, not shared. Processes can communicate with one another by passing messages. They can broadcast, send, receive, and synchronize their operations. After all computations are done, all data are aggregated to form a final result.

2. Matrix is read from a text file and then stored as a structure variable from the Matrix struct

    Representing matrix as an array within a struct can reduce memory access time. Structs have contiguous memory layout while std::unique_ptr from standard library have a scattered memory access incurring additional overheads. 

3. The dimension information of the matrix are broadcasted

    The first rank gets the information of the matrix dimension and broadcast it to all other processes. The dimension information is needed to calculate the chunks and other matrix operation by each processes. 

4. Scatter the matrix into chunks

    The matrix then scattered into several chunks. The scattering is cyclic, using row wise distribution, meaning the first row will be computed by the first rank, the second row by the second rank, and (n + 1) row by n rank where n is the number of processes. This ensures that each process calculates a small amount of data (1 row of data) before passing it to another process to start its computation, as if the process does not calculate anything at all, thus all processes start at the same time.

5. Partition each row computation

    Partition row computation (unit matrix transformation and eliminations) to be performed by its corresponding process. 

6. Broadcast the pivot
  
    After transforming all chunks row into a unit matrix (value of pivot is 1), the row, we called it the pivot row, has to be broadcasted to every other processes. This pivot row will be used as a base for forward and backward elimination.

7. Perform the eliminations

    The chunk elements that are not a pivot have to be 0 to forn row-echelon matrix. The elimination is performed by negating the elements with a ratio of the elements and the unit pivot, eventually cancelling the elements.

8. Aggregate the result from each processes

    The result which is a row-echelon form of the initial chunks are gathered from each processes to get the final result before printing it into an output text file.


### Built With

* [![Cpp][Cpp.cpp]][Cpp-url]
* [Open MPI][OpenMPI-url]

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
3. Run the program `mpi.cpp` using makefile
  <br/>

   ```sh
   make mpi
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
    make mpi
    ```
4. The result is stored inside `result` folder as `mpi_8.txt`
    <br/>

    ```ssh
    /* mpi_8.txt */

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
[Cpp.cpp]: https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white
[Cpp-url]: https://isocpp.org/std/the-standard
[OpenMPI-url]: https://github.com/open-mpi