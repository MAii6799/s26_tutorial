#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>

std::atomic<int> count(0); // Use atomic variable for thread-safe increment

std::mutex mtx;

void plus(){
    for (int i=0;i<100000;i++){
        // mtx.lock();
        count++;
        // mtx.unlock();
    }
}


int main(){
    std::thread t1(plus);
    std::thread t2(plus);

    t1.join();
    t2.join();

    std::cout << "count = " << count << std::endl;

    return 0;
}