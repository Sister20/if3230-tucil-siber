<!-- LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Sister20/if3230-tucil-siber">
    <img src="../../public/open-mp-ic.png" alt="Logo" height="80">
  </a>

  <h3 align="center">TUCIL 2 - Open MP</h3>

  <p align="center">
    Matrix inverse solver using shared memory.
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

Tucil 2 of Parallel and Distributed Systems (IF3230). Matrix inverse solver using shared memory. 

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
        </tr>
        <tr>
            <th scope="col">Serial</th>
            <th scope="col">Open MPI</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>32</td>
            <td>00.02</td>
            <td>00.31</td>
        </tr>
        <tr>
            <td>64</td>
            <td>00.04</td>
            <td>00.32</td>
        </tr>
        <tr>
            <td>128</td>
            <td>00.12</td>
            <td>00.33</td>
        </tr>
        <tr>
            <td>256</td>
            <td>00.51</td>
            <td>00.4</td>
        </tr>
        <tr>
            <td>512</td>
            <td>02.54</td>
            <td>00.79</td>
        </tr>
        <tr>
            <td>1024</td>
            <td>13.65</td>
            <td>02.72</td>
        </tr>
        <tr>
            <td>2048</td>
            <td>86.15</td>
            <td>16.85</td>
        </tr>
    </tbody>
</table>

### Methods

1. Matrix is read from a text file and then stored as an array of double

2. For each

### Built With

* [![Cpp][Cpp.cpp]][Cpp-url]
* [Open MP][OpenMP-url]

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites
* Linux operating system or WSL
* G++ for C++ compilation
* Make to run makefile
* Open MP library

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
3. Run the program `mp.cpp` using makefile
  <br/>

   ```sh
   make mp
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
    make mp
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
[OpenMP-url]: https://www.openmp.org/spec-html/5.0/openmpse14.html