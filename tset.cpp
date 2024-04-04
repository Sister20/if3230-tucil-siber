//
// Created by lenovo on 04/04/2024.
//
#include <iostream>
using namespace std;

int main (){

    long sum = 0;
//#pragma omp parallel for
    for (long i = 0; i < 100000; i++) {
        for(int j = 0; j < 1000; j++) sum += i;
        for(int j = 0; j < 1000; j++) sum += i;
        for(int j = 0; j < 1000; j++) sum += i;
    }
    printf("Sum = %ld\n", sum);
    return 0;
}
