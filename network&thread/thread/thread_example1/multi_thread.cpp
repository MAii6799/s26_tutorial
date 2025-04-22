#include <iostream>
#include <thread>
#include <mutex>

int id = 0;
std::mutex mtx;
void print_message(int i) {
    while(1){
        // mtx.lock();
        id = i;
        std::cout <<"thread "<<i<<" Speak:"<< "Hello from thread " << id << std::endl;
        // mtx.unlock();
    }
}
    

int main() {
    
    //开两个线程执行print_message函数
    std::thread t1(print_message,1);
    std::thread t2(print_message,2);

    //等待线程结束
    t1.join();
    t2.join();

    std::cout << "Main thread finished." << std::endl;
    return 0;
}