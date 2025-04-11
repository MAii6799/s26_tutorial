# ifndef TRANSFORM_TEACH
# define TRANSFORM_TEACH
# include <iostream>
# include <vector>
#include <algorithm>
#include <cmath>

// 前向声明Euler类
class Euler;

// use transform sequence 
class Quaternion {
    private :
        
    public :
        double w, x, y, z;
        Quaternion(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}
        Quaternion operator*(const Quaternion& q) const {
            return Quaternion(
                w * q.w - x * q.x - y * q.y - z * q.z,
                w * q.x + x * q.w + y * q.z - z * q.y,
                w * q.y - x * q.z + y * q.w + z * q.x,
                w * q.z + x * q.y - y * q.x + z * q.w
            );
        }
        void print() const {
            std::cout << "Quaternion: (" << w << ", " << x << ", " << y << ", " << z << ")" << std::endl;
        }
        Quaternion conjugate() const {
            return Quaternion(w, -x, -y, -z);
        }
        Quaternion inverse() const {
            double norm = w * w + x * x + y * y + z * z;
            return Quaternion(w / norm, -x / norm, -y / norm, -z / norm);
        }
        std::vector<double> to_vector() const {
            return {w, x, y, z};
        }
        
        // 声明方法但在类外定义实现
        Quaternion from_euler(const Euler& euler);
        
        Euler to_euler() const;
    };

class Euler {
    private :
        
    public :
        double roll, pitch, yaw;
        Euler(double roll, double pitch, double yaw) : roll(roll), pitch(pitch), yaw(yaw) {}
        Euler(const Quaternion& q);
        
        Euler operator+(const Euler& q) const {
            return Euler(roll + q.roll, pitch + q.pitch, yaw + q.yaw);
        }
        Euler operator-(const Euler& q) const {
            return Euler(roll - q.roll, pitch - q.pitch, yaw - q.yaw);
        }
        Euler operator*(const Euler& q) const {
            return Euler(
                roll * q.roll - pitch * q.pitch - yaw * q.yaw,
                roll * q.pitch + pitch * q.roll + yaw * q.yaw,
                roll * q.yaw - pitch * q.yaw + yaw * q.roll + pitch * q.pitch
            );
        }
        void print() const {
            std::cout << "Euler: (" << roll << ", " << pitch << ", " << yaw << ")" << std::endl;
        }
        Euler conjugate() const {
            return Euler(roll, -pitch, -yaw);
        }
        Euler inverse() const {
            double norm = roll * roll + pitch * pitch + yaw * yaw;
            return Euler(roll / norm, -pitch / norm, -yaw / norm);
        }
        std::vector<double> to_vector() const {
            return {roll, pitch, yaw};
        }

        Quaternion to_quaternion() const;
        Euler from_quaternion(const Quaternion& q);
    };


// 在两个类声明后实现相互依赖的方法
inline Quaternion Quaternion::from_euler(const Euler& euler) {
    double cy = cos(euler.yaw * 0.5);
    double sy = sin(euler.yaw * 0.5);
    double cp = cos(euler.pitch * 0.5);
    double sp = sin(euler.pitch * 0.5);
    double cr = cos(euler.roll * 0.5);
    double sr = sin(euler.roll * 0.5);

    // using rotation sequence: zyx

    return Quaternion(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    );
}

inline Euler Quaternion::to_euler() const {
    double t0 = 2.0 * (w * x + y * z);
    double t1 = 1.0 - 2.0 * (x * x + y * y);
    double roll_x = atan2(t0, t1);

    double t2 = 2.0 * (w * y - z * x);
    t2 = (t2 > 1.0) ? 1.0 : t2;
    t2 = (t2 < -1.0) ? -1.0 : t2;
    double pitch_y = asin(t2);

    double t3 = 2.0 * (w * z + x * y);
    double t4 = 1.0 - 2.0 * (y * y + z * z);
    double yaw_z = atan2(t3, t4);

    return Euler(roll_x, pitch_y, yaw_z);
}

inline Euler::Euler(const Quaternion& q) {
    double t0 = 2.0 * (q.w * q.x + q.y * q.z);
    double t1 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    roll = atan2(t0, t1);

    double t2 = 2.0 * (q.w * q.y - q.z * q.x);
    t2 = (t2 > 1.0) ? 1.0 : t2;
    t2 = (t2 < -1.0) ? -1.0 : t2;
    pitch = asin(t2);

    double t3 = 2.0 * (q.w * q.z + q.x * q.y);
    double t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    yaw = atan2(t3, t4);
}

inline Quaternion Euler::to_quaternion() const {
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    // using rotation sequence: zyx

    return Quaternion(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    );
}

inline Euler Euler::from_quaternion(const Quaternion& q) {
    double t0 = 2.0 * (q.w * q.x + q.y * q.z);
    double t1 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    double roll_x = atan2(t0, t1);

    double t2 = 2.0 * (q.w * q.y - q.z * q.x);
    t2 = (t2 > 1.0) ? 1.0 : t2;
    t2 = (t2 < -1.0) ? -1.0 : t2;
    double pitch_y = asin(t2);

    double t3 = 2.0 * (q.w * q.z + q.x * q.y);
    double t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    double yaw_z = atan2(t3, t4);

    return Euler(roll_x, pitch_y, yaw_z);
}


class Transform {
    public :
        std::string name;
        std::string parent;
        std::vector<std::string> children;
        Quaternion quaternion=Quaternion(1, 0, 0, 0);
        int dx=0, dy=0, dz=0;

        Transform(const std::string& name, const std::string& parent, const Quaternion& quaternion, int dx, int dy, int dz)
            : name(name), parent(parent), quaternion(quaternion), dx(dx), dy(dy), dz(dz) {}
        Transform(const std::string& name, const std::string& parent, const Quaternion& quaternion)
            : name(name), parent(parent), quaternion(quaternion) {}
        Transform(const std::string& name, const std::string& parent)
            : name(name), parent(parent) {}
        Transform(const std::string& name)
            : name(name) {}
        Transform() : name(""), parent("") {}
        Transform(const Transform& other)
            : name(other.name), parent(other.parent), children(other.children), quaternion(other.quaternion), dx(other.dx), dy(other.dy), dz(other.dz) {}
        Transform& operator=(const Transform& other) {
            if (this != &other) {
                name = other.name;
                parent = other.parent;
                children = other.children;
                quaternion = other.quaternion;
                dx = other.dx;
                dy = other.dy;
                dz = other.dz;
            }
            return *this;
        }
        Transform(Transform&& other) noexcept
            : name(std::move(other.name)), parent(std::move(other.parent)), children(std::move(other.children)), quaternion(std::move(other.quaternion)), dx(other.dx), dy(other.dy), dz(other.dz) {}
        Transform& operator=(Transform&& other) noexcept {
            if (this != &other) {
                name = std::move(other.name);
                parent = std::move(other.parent);
                children = std::move(other.children);
                quaternion = std::move(other.quaternion);
                dx = other.dx;
                dy = other.dy;
                dz = other.dz;
            }
            return *this;
        }
        bool operator==(const Transform& other) const {
            return name == other.name && parent == other.parent && children == other.children && quaternion.to_vector() == other.quaternion.to_vector() && dx == other.dx && dy == other.dy && dz == other.dz;
        }
        bool operator!=(const Transform& other) const {
            return !(*this == other);
        }
        void add_child(const std::string& child_name) {
            children.push_back(child_name);
        }
        void remove_child(const std::string& child_name) {
            children.erase(std::remove(children.begin(), children.end(), child_name), children.end());
        }
        void print() const {
            std::cout << "Transform: " << name << ", Parent: " << parent << ", Children: ";
            for (const auto& child : children) {
                std::cout << child << " ";
            }
            std::cout << std::endl;
            quaternion.print();
        }
        std::vector<double> to_vector() const {
            return quaternion.to_vector();
        }
        Quaternion to_quaternion() const {
            return quaternion;
        }
        Euler to_euler() const {
            return quaternion.to_euler();
        }
};

class TransformManager {
    private :
        std::vector<Transform> transforms;
        bool path_finder(const std::string& src, const std::string& dst, std::vector<std::string>& path) const {
            // use dfs to find path
            std::vector<std::string> stack;
            stack.push_back(src);
            while (!stack.empty()) {
                std::string current = stack.back();
                stack.pop_back();
                if (current == dst) {
                    path.push_back(current);
                    return true;
                }
                for (const auto& transform : transforms) {
                    if (transform.parent == current) {
                        stack.push_back(transform.name);
                    }
                }
            }
            // 如果没有找到路径，返回空路径
            path.clear();
            std::cerr << "No path found from " << src << " to " << dst << std::endl;
            return false;
        }
    public :
        void add_transform(const Transform& transform) {
            transforms.push_back(transform);
        }
        void remove_transform(const std::string& name) {
            transforms.erase(std::remove_if(transforms.begin(), transforms.end(), [&](const Transform& t) { return t.name == name; }), transforms.end());
        }
        Transform get_transform(const std::string& src, const std::string& dst) const {

        }
};
#endif