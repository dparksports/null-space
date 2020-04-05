#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>

using namespace std;
using namespace Eigen;

void print(const string name, const Eigen::MatrixXf m) {
    std::cout << name << ": ( " <<m.rows()  << ", " << m.cols() << " ) = " << std::endl << m << std::endl << std::endl;
}

int main() {
    Eigen::MatrixXf pts2d(6,2), pts3d(6,3);
    Eigen::MatrixXf eigenPts2dA(20,2), eigenPts2dB(20,2);

    eigenPts2dA <<
                880, 214,
            43,  203,
            270,  197,
            886,  347,
            745,  302,
            943,  128,
            476,  590,
            419,  214,
            317,  335,
            783,  521,
            235,  427,
            665,  429,
            655,  362,
            427,  333,
            412,  415,
            746,  351,
            434,  415,
            525,  234,
            716,  308,
            602,  187;
    print("eigenPts2dA", eigenPts2dA);

    eigenPts2dB <<
    731,  238,
    22,   248,
    204,  230,
    903,  342,
    635,  316,
    867,  177,
    958,  572,
    328,  244,
    426,  386,
    1064, 470,
    480,  495,
    964,  419,
    695,  374,
    505,  372,
    645,  452,
    692,  359,
    712,  444,
    465,  263,
    591,  324,
    447,  213;
    print("eigenPts2dB", eigenPts2dB);

    assert(eigenPts2dA.cols() == eigenPts2dB.cols() &&
    eigenPts2dA.rows() == eigenPts2dB.rows() &&
    eigenPts2dA.cols() == 2);

    const size_t rows = eigenPts2dB.rows();
    const size_t cols = 8;
    Eigen::MatrixXf A(rows, cols);
    Eigen::MatrixXf b = Eigen::MatrixXf::Constant(rows, 1, -1);

    for (int row = 0; row < rows; ++row) {
        float u = eigenPts2dA(row, 0);
        float v = eigenPts2dA(row, 1);

        float u_p = eigenPts2dB(row, 0);
        float v_p = eigenPts2dB(row, 1);

        A.row(row) << u_p * u, u_p * v, u_p, v_p * u, v_p * v, v_p, u, v;
    }
    print("A", A);
    print("b", b);

    Eigen::MatrixXf AtA = (A.transpose() * A);
    Eigen::MatrixXf Atb = (A.transpose() * b);
    Eigen::MatrixXf leastSquare = AtA.ldlt().solve(Atb);
    print("AtA", AtA);
    print("Atb", Atb);
    print("leastSquare", leastSquare);

    leastSquare.conservativeResize(leastSquare.rows() + 1, leastSquare.cols());
    leastSquare(leastSquare.rows() - 1, 0) = 1;
    print("leastSquare", leastSquare);

    return 0;
}
