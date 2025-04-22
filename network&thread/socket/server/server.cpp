#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>

int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        std::cerr << "socket error\n";
        return 1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8888);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) == -1) {
        std::cerr << "bind error\n";
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 1) == -1) {
        std::cerr << "listen error\n";
        close(server_fd);
        return 1;
    }

    std::cout << "Server listening on port 8888...\n";
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(server_fd, (sockaddr*)&client_addr, &client_len);
    if (client_fd == -1) {
        std::cerr << "accept error\n";
        close(server_fd);
        return 1;
    }

    char buf[1024];
    while (true) {
        ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) break;
        buf[n] = '\0';
        std::cout << "Received: " << buf << std::endl;
    }

    close(client_fd);
    close(server_fd);
    return 0;
}