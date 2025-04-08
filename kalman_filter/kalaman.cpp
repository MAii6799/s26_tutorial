#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <fstream>

class KalmanFilter {
    // basic Kalman Filter
public:
    KalmanFilter(int state_dim, int measurement_dim)
        : state_dim(state_dim), measurement_dim(measurement_dim) {
        x = Eigen::VectorXd::Zero(state_dim); // State vector
        P = Eigen::MatrixXd::Identity(state_dim, state_dim); // State covariance
        F = Eigen::MatrixXd::Identity(state_dim, state_dim); // State transition matrix
        H = Eigen::MatrixXd::Zero(measurement_dim, state_dim); // Measurement matrix
        R = Eigen::MatrixXd::Identity(measurement_dim, measurement_dim); // Measurement noise covariance
        Q = Eigen::MatrixXd::Identity(state_dim, state_dim); // Process noise covariance
    }
    void predict() {
        // Predict the state
        x = F * x;
        P = F * P * F.transpose() + Q;
    }
    void update(const Eigen::VectorXd &z) {
        // Update the state with the measurement
        Eigen::VectorXd y = z - H * x; // Measurement residual
        Eigen::MatrixXd S = H * P * H.transpose() + R; // Residual covariance
        Eigen::MatrixXd K = P * H.transpose() * S.inverse(); // Kalman gain
        x += K * y; // Update state estimate
        P = (Eigen::MatrixXd::Identity(state_dim, state_dim) - K * H) * P; // Update covariance
    }
    void setTransitionMatrix(const Eigen::MatrixXd &F) {
        this->F = F;
    }
    void setMeasurementMatrix(const Eigen::MatrixXd &H) {
        this->H = H;
    }
    void setProcessNoiseCovariance(const Eigen::MatrixXd &Q) {
        this->Q = Q;
    }
    void setMeasurementNoiseCovariance(const Eigen::MatrixXd &R) {
        this->R = R;
    }
    void setInitialState(const Eigen::VectorXd &x0) {
        this->x = x0;
    }
    void setInitialCovariance(const Eigen::MatrixXd &P0) {
        this->P = P0;
    }
    Eigen::VectorXd getState() const {
        return x;
    }
    Eigen::MatrixXd getCovariance() const {
        return P;
    }
    void setStateDimension(int state_dim) {
        this->state_dim = state_dim;
    }
    void setMeasurementDimension(int measurement_dim) {
        this->measurement_dim = measurement_dim;
    }
private:
    int state_dim; // Dimension of the state vector
    int measurement_dim; // Dimension of the measurement vector
    Eigen::VectorXd x; // State vector
    Eigen::MatrixXd P; // State covariance matrix
    Eigen::MatrixXd F; // State transition matrix
    Eigen::MatrixXd H; // Measurement matrix
    Eigen::MatrixXd R; // Measurement noise covariance matrix
    Eigen::MatrixXd Q; // Process noise covariance matrix
};

int main(){
    KalmanFilter kf(2, 1); // 2D state, 1D measurement
    Eigen::MatrixXd F(2, 2);
    F << 1, 1,
         0, 1; // State transition matrix
    Eigen::MatrixXd H(1, 2);
    H << 1, 0; // Measurement matrix
    Eigen::MatrixXd Q(2, 2);
    double q = 0.0000001;
    Q << q, 0,
         0, q; // Process noise covariance
    Eigen::MatrixXd R(1, 1);
    double r = 100;
    R << r; // Measurement noise covariance
    kf.setTransitionMatrix(F);
    kf.setMeasurementMatrix(H);
    kf.setProcessNoiseCovariance(Q);
    kf.setMeasurementNoiseCovariance(R);

    // read data from file
    std::ifstream file("../homework_data_4.txt");
    // file style: time data
    // 0.0 0.0
    std::string line;
    std::vector<double> time;
    std::vector<double> data;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double t, d;
        if (!(iss >> t >> d)) { break; } // error
        time.push_back(t);
        data.push_back(d);
    }
    file.close();
    // set initial state
    Eigen::VectorXd x0(2);
    x0 << data[0], 0; // initial position and velocity
    kf.setInitialState(x0);
    Eigen::MatrixXd P0(2, 2);
    P0 << 1, 0,
          0, 1; // initial covariance
    kf.setInitialCovariance(P0);
    // run Kalman filter
    std::vector<Eigen::VectorXd> states;
    for (size_t i = 0; i < time.size(); ++i) {
        kf.predict();
        Eigen::VectorXd z(1);
        z << data[i]; // measurement
        kf.update(z);
        states.push_back(kf.getState());
    }
    // write data to file
    std::ofstream output_file("kalman_output.txt");
    for (size_t i = 0; i < states.size(); ++i) {
        output_file << time[i] << " " << states[i][0] << std::endl;
    }
    output_file.close();
    std::cout << "Kalman filter output saved to kalman_output.txt" << std::endl;
    return 0;
}