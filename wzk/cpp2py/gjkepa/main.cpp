#include "Volume.h"
#include "DistanceComputationState.h"

#include <iostream>
#include <ctime>
#include <stack>

std::stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    std::cout << "Time elapsed: "
              << ((float)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << std::endl;
    tictoc_stack.pop();
}


#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>
#include <array>

using namespace std;

template <class T>
void print(T& c){
    for(auto i = c.begin(); i != c.end(); i++ ){
        std::cout << *i << endl;
    }
}

/*
int main2( ){
    float a[] = {1, 1.3, 1.5, 0.9, 0.1, 0.2};
    std::array<float, 3> b = {1, 2, 3};
    std::vector<float> data(a, a+sizeof(a) / (sizeof(a[0])));
    std::vector<std::array<float, 3>> data2;
    std::vector<std::array<float, 3>> data3(a, a+sizeof(a) / (3*sizeof(a[0])));

    data2.push_back(b);

    //cout << sizeof(a) << "\n";
    //cout << sizeof(a[0]) << "\n";
    cout << data.size() << " ELEMENTS\n";
    print( data  );
    print( data2[0]  );

}
*/


int main() {
    std::cout << "Hello, World!" << std::endl;

    std::vector<Vector> vA, vB, vC;
    vA.emplace_back(1, 0, 2);
    vA.emplace_back(2, 0, 2);
    vA.emplace_back(1, 1, 2);
    vA.emplace_back(2, 1, 2);

    vB.emplace_back(3, 0, 0);
    vB.emplace_back(4, 0, 0);

    vC.emplace_back(6, 0, 1);
    vC.emplace_back(5, 0, 1);
    vC.emplace_back(7, 0, 1);
    vC.emplace_back(8, 0, 1);
    vC.emplace_back(9, 0, 1);

    Volume volA = Volume(vA, 0.1);
    Volume volB = Volume(vB, 0.2);
    Volume volC = Volume(vC, 0.3);

    DistanceComputationState dcs, dcs2;
    tic();
    dcs.compute(volA, volB, 100, 100);
    toc();
    std::cout << dcs.pointOnA << std::endl;
    std::cout << dcs.pointOnB << std::endl;
    std::cout << dcs.distanceBound << std::endl;

    tic();
    dcs.compute(volA, volC, 100, 10);
    toc();

    tic();
    dcs2.compute(volA, volC, 100, 10);
    toc();

    std::cout << dcs.pointOnA << std::endl;
    std::cout << dcs.pointOnB << std::endl;
    std::cout << dcs.distanceBound << std::endl;

    return 0;

}
