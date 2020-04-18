
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

// cv::line?
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <opencv2/core/mat.hpp>
// cv:imread
#include <opencv2/highgui/highgui.hpp>

// setprecision
#include <iomanip>
#include <iostream>

// logger
//#include <spdlog/fmt/ostr.h>
//#include <spdlog/spdlog.h>

using namespace std;
using namespace Eigen;

// Struct to allow prettier printing of cv::Mats
struct FormattedMat {
    cv::Mat _mat;

    FormattedMat(const cv::Mat& mat) : _mat(mat) {}

    friend std::ostream& operator<<(std::ostream& stream, const FormattedMat& fm) {
        std::ios init(NULL);
        init.copyfmt(stream);
        // Align right
        stream << "[" << std::setprecision(5);
        for (size_t y = 0; y < fm._mat.rows; y++) {
            stream << (y == 0 ? " " : "  ");
            for (size_t x = 0; x < fm._mat.cols; x++) {
                stream << std::left << std::setw(13) << fm._mat.at<float>(y, x);
            }
            stream << (y < fm._mat.rows - 1 ? "\n" : " ");
        }
        stream << "]";
        // Restore original formatting
        stream.copyfmt(init);
        return stream;
    }
};

void print(const string name, const cv::Mat m) {
    std::cout << name << ": ( " << m.size() << " ) = " << std::endl << FormattedMat(m) << std::endl << std::endl;
//    auto logger = spdlog::get(config::STDOUT_LOGGER);
//    logger->info(name, ":\n{}", FormattedMat(fMatEst));
}

void print(const string name, const Eigen::MatrixXf m) {
    std::cout << name << ": ( " << m.rows()  << ", " << m.cols() << " ) = " << std::endl << m << std::endl << std::endl;
}

/**
 * \brief Draw epipolar lines on a given image
 *
 * \param img Input/output image on which the lines will be drawn
 * \param epiLines Set of epipolar lines to draw.
 * \param color Color in which the lines should be drawn.
 */
void drawEpipolarLines(cv::Mat& img, const cv::Mat& epiLines, const cv::Scalar color) {
    size_t rows = img.rows;
    size_t cols = img.cols;

    // Compute left and right edge intersections for the image
    cv::Mat P_UL = (cv::Mat_<float>(3, 1) << 0, 0, 1);               // Upper left
    cv::Mat P_BL = (cv::Mat_<float>(3, 1) << 0, rows - 1, 1);        // Bottom left
    cv::Mat P_UR = (cv::Mat_<float>(3, 1) << cols - 1, 0, 1);        // Upper right
    cv::Mat P_BR = (cv::Mat_<float>(3, 1) << cols - 1, rows - 1, 1); // Bottom right

    // Compute the lines corresponding to the left and right edges of the image
    cv::Mat I_L = P_UL.cross(P_BL);
    cv::Mat I_R = P_UR.cross(P_BR);

    // Iterate over columns of lines and compute the endpoints, and then draw them
    for (int col = 0; col < epiLines.cols; col++) {
        cv::Mat curLine = epiLines.col(col);
        cv::Mat P_iL = curLine.cross(I_L);
        cv::Mat P_iR = curLine.cross(I_R);

        // Scale correctly
        P_iL /= P_iL.at<float>(2, 0);
        P_iR /= P_iR.at<float>(2, 0);

        // Transpose just for logging (not actually used in calculation)
        cv::Mat P_iL_T, P_iR_T;
        cv::transpose(P_iL, P_iL_T);
        cv::transpose(P_iR, P_iR_T);

        cv::line(img,
                 cv::Point2f(P_iL.at<float>(0, 0), P_iL.at<float>(1, 0)),
                 cv::Point2f(P_iR.at<float>(0, 0), P_iR.at<float>(1, 0)),
                 color);
    }
}


void fundamental() {
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

    cv::Mat cvLeastSquare;
    cv::eigen2cv(leastSquare, cvLeastSquare);
    cv::Mat estimatedF = cvLeastSquare.reshape(0, 3);
    print("cvLeastSquare", cvLeastSquare);
    print("estimatedF", estimatedF);

    cv::cv2eigen(estimatedF, leastSquare);
    assert(leastSquare.cols() == 3 && leastSquare.rows() == 3);
    auto svd = leastSquare.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto singularValues = svd.singularValues();
    print("Diagonal", singularValues.asDiagonal());

    // set smallest eigenvalue to zero, since rank 2
    singularValues(singularValues.rows() - 1, singularValues.cols() -1) = 0;
    Eigen::MatrixXf dHat = singularValues.asDiagonal();
    print("dHat", singularValues.asDiagonal());

    Eigen::MatrixXf fHat = svd.matrixU() * dHat * svd.matrixV().transpose();
    print("fHat", fHat);

    cv::Mat fHatcv;
    cv::eigen2cv(fHat, fHatcv);
    print("fHatcv", fHatcv);

    cv::Mat pointsAcv, pointsBcv;
    cv::eigen2cv(eigenPts2dA, pointsAcv);
    cv::eigen2cv(eigenPts2dB, pointsBcv);

    cv::Mat pointsA_T, pointsB_T;
    cv::transpose(pointsAcv, pointsA_T);
    cv::transpose(pointsBcv, pointsB_T);
    print("pointsA_T", pointsA_T);
    print("pointsB_T", pointsB_T);

    // Append a row of ones to the input points to make 3xn matrices
    pointsA_T.push_back(cv::Mat::ones(1, pointsA_T.cols, pointsA_T.type()));
    pointsB_T.push_back(cv::Mat::ones(1, pointsB_T.cols, pointsB_T.type()));
    print("pointsA_T", pointsA_T);
    print("pointsB_T", pointsB_T);

    cv::transpose(pointsA_T, pointsAcv);
    cv::transpose(pointsB_T, pointsBcv);
    print("pointsAcv", pointsAcv);
    print("pointsBcv", pointsBcv);

    cv::Mat linesA = pointsBcv * fHatcv;
    print("linesA", linesA);

    cv::Mat linesA_T, linesB_T;
    cv::transpose(linesA, linesA_T);
    print("linesA_T", linesA_T);
    cv::Mat linesB = fHatcv * pointsA_T;
    print("linesB", linesB);

    std::string _outputPathPrefix = "../";
    cv::Mat picA = cv::imread(_outputPathPrefix + "pic_a.jpg", cv::IMREAD_UNCHANGED);
    cv::Mat picB = cv::imread(_outputPathPrefix + "pic_b.jpg", cv::IMREAD_UNCHANGED);

    // Draw epipolar lines on A and B images
    drawEpipolarLines(picA, linesA_T, CV_RGB(0, 0xFF, 0));
    drawEpipolarLines(picB, linesB, CV_RGB(0, 0xFF, 0));

    cv::imwrite(_outputPathPrefix + "ps3-2-c-1-rank2.png", picA);
    cv::imwrite(_outputPathPrefix + "ps3-2-c-2-rank2.png", picB);

}

//using namespace cv;

int main() {
//    fundamental();

    string filepath = "../simA.jpg";
    cv::Mat  src = imread( filepath, cv::IMREAD_COLOR );

    cv::Mat src_gray;
    cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    int ksize = 1;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    cv::Mat  gradientX, gradientY;
    cv::Sobel(src_gray, gradientX, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(src_gray, gradientY, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

    cv::Mat gradCombined(gradientX.rows + gradientY.rows, gradientX.cols, CV_8UC1);
    cv::Mat gradXNorm(gradientX.rows, gradientX.cols, CV_8UC1);
    cv::Mat gradYNorm(gradientY.rows, gradientY.cols, CV_8UC1);

    cv::normalize(gradientY, gradXNorm, 0, 255, cv::NORM_MINMAX);
    cv::normalize(gradientY, gradYNorm, 0, 255, cv::NORM_MINMAX);
    cv::hconcat(gradXNorm, gradYNorm, gradCombined);

    string outputPath = "../simA-sobel.jpg";
    cv::imwrite(outputPath, gradCombined);
}
