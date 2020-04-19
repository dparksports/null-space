
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

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

void printRow(const string name, const cv::Mat m) {
    std::cout << name << ": (" << m.size()  <<  "), type:" <<   type2str(m.type()) << " = " << std::endl << m.row(0) << std::endl << std::endl;
}

void print(const string name, const cv::Mat m) {
    std::cout << name << ": ( " << m.size()  <<  " ), type: " <<   m.type() << " = " << std::endl << m << std::endl << std::endl;
//    std::cout << name << ": ( " << m.size()  <<  " ), type: " <<   m.type() << " = " << std::endl << FormattedMat(m) << std::endl << std::endl;
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

void getCornerResponse(const cv::Mat& gradX,
                                    const cv::Mat& gradY,
                                    const size_t windowSize,
                                    const double gaussianSigma,
                                    const float harrisScore,
                                    cv::Mat& cornerResponse) {
    assert(gradX.rows == gradY.rows && gradX.cols == gradY.cols && gradX.type() == CV_32F &&
           gradX.type() == gradY.type());
    assert(windowSize % 2 == 1);

    cornerResponse = cv::Mat::zeros(gradX.rows, gradX.cols, CV_32F);

    // Get a 1D Gaussian kernel with a given size and sigma
    cv::Mat gauss = cv::getGaussianKernel(windowSize, gaussianSigma, gradX.type());
    // Outer product for a 2D matrix
    gauss = gauss * gauss.t();
    // Iterate over each pixel in the image and compute the second moment matrix for each pixel,
    // where the weights are the Gaussian kernel
    int windowRad = windowSize / 2;
    for (int y = 0; y < gradX.rows; y++) {
        for (int x = 0; x < gradX.cols; x++) {
            cv::Mat secondMoment = cv::Mat::zeros(2, 2, CV_32F);
            for (int wy = -windowRad; wy <= windowRad; wy++) {
                for (int wx = -windowRad; wx <= windowRad; wx++) {
                    // Get the gradient values
                    float gradXVal = gradX.at<float>(std::min(std::max(0, y + wy), gradX.rows - 1),
                                                     std::min(std::max(0, x + wx), gradX.cols - 1));
                    float gradYVal = gradY.at<float>(std::min(std::max(0, y + wy), gradY.rows - 1),
                                                     std::min(std::max(0, x + wx), gradY.cols - 1));

                    float weight = gauss.at<float>(wy + windowRad, wx + windowRad);

                    // Build up gradient matrix
                    cv::Mat gradVals = (cv::Mat_<float>(2, 2) << gradXVal * gradXVal,
                            gradXVal * gradYVal,
                            gradXVal * gradYVal,
                            gradYVal * gradYVal);

                    // Add to second moment matrix sum
                    secondMoment = secondMoment + weight * gradVals;
                }
            }
            // Compute the corner response value, R
            float trace = (cv::trace(secondMoment))[0];
            float R = cv::determinant(secondMoment) - harrisScore * trace * trace;

            cornerResponse.at<float>(y, x) = R;
        }
    }
}

void drawDots(const cv::Mat& mask, const cv::Mat& img, cv::Mat& dottedImg) {
    // Convert the image to rgb
    cv::Mat img3Ch;
    img.convertTo(img3Ch, CV_8UC3);
    dottedImg.create(img3Ch.rows, img3Ch.cols, img3Ch.type());
    cv::cvtColor(img, dottedImg, cv::COLOR_GRAY2RGB);

    cv::Mat maskNorm(mask.rows, mask.cols, CV_8U);
    cv::normalize(mask, maskNorm, 0, 255, cv::NORM_MINMAX, CV_8U);
    dottedImg.setTo(cv::Scalar(0, 0, 255), maskNorm);
}

void computeHarrisResponse(string current_path, string filename) {
    string imgPath = current_path + filename + ".jpg";

    cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    printRow("img", img);

    cv::Mat input;
    img.convertTo(input, CV_32F);
    printRow("input", input);

    int sobel_kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;

    cv::Mat  gradientX, gradientY;
    cv::Sobel(input, gradientX, ddepth, 1, 0, sobel_kernel_size, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(input, gradientY, ddepth, 0, 1, sobel_kernel_size, scale, delta, cv::BORDER_DEFAULT);
    printRow("gradientX", gradientX);
    printRow("gradientY", gradientY);

    cv::Mat gradCombined(gradientX.rows + gradientY.rows, gradientX.cols, CV_8UC1);
    cv::Mat gradXNorm(gradientX.rows, gradientX.cols, CV_8UC1);
    cv::Mat gradYNorm(gradientY.rows, gradientY.cols, CV_8UC1);

    cv::normalize(gradientX, gradXNorm, 0, 255, cv::NORM_MINMAX);
    cv::normalize(gradientY, gradYNorm, 0, 255, cv::NORM_MINMAX);
    cv::hconcat(gradXNorm, gradYNorm, gradCombined);
    printRow("gradXNorm", gradXNorm);
    printRow("gradYNorm", gradYNorm);

    string outputPath = current_path + filename + "-sobel.jpg";
    cv::imwrite(outputPath, gradCombined);

    cv::Mat cornerResponse = cv::Mat::zeros(gradientX.rows, gradientX.cols, CV_32F);
    printRow("cornerResponse", cornerResponse);

    const size_t windowSize = 5;
    const double sigmaGaussian = 1.5;
    const float alpha = 0.04;
    cv::Mat kernelGaussian2D = cv::getGaussianKernel(windowSize, sigmaGaussian, gradientX.type());
    print("kernelGaussian2D", kernelGaussian2D);
    kernelGaussian2D = kernelGaussian2D * kernelGaussian2D.t(); // generate a 2d matrix with an outer product
    print("kernelGaussian1D", kernelGaussian2D);

    // compute a second moment matrix for each pixel, where the weights are the Gaussian kernel
    int windowHalf = windowSize / 2;
    for (int row = 0; row < gradientX.rows; row++) {
        for (int col = 0; col < gradientX.cols; col++) {
            cv::Mat secondMomentMatrix = cv::Mat::zeros(2, 2, CV_32F);
//            print("secondMomentMatrix", secondMomentMatrix);

            for (int rowWeight = -windowHalf; rowWeight <= windowHalf; rowWeight++) {
                for (int colWeight = -windowHalf; colWeight <= windowHalf; colWeight++) {

                    float xGradientAt = gradientX.at<float>(std::min(std::max(0, row + rowWeight), gradientX.rows - 1),
                                                            std::min(std::max(0, col + colWeight), gradientX.cols - 1));

                    float yGradientAt = gradientY.at<float>(std::min(std::max(0, row + rowWeight), gradientY.rows - 1),
                                                            std::min(std::max(0, col + colWeight), gradientY.cols - 1));

                    // a gradient matrix
                    cv::Mat gradientMatrix = (cv::Mat_<float>(2, 2)  <<
                            xGradientAt * xGradientAt,
                            xGradientAt * yGradientAt,
                            xGradientAt * yGradientAt,
                            yGradientAt * yGradientAt);
//                    print("gradientMatrix", gradientMatrix);

                    float weightXY = kernelGaussian2D.at<float>(rowWeight + windowHalf, colWeight + windowHalf);
                    secondMomentMatrix = secondMomentMatrix + weightXY * gradientMatrix;
//                    print("secondMomentMatrix", secondMomentMatrix);
                }
            }

            // compute Harris corner response
            // R = det(M) - alpha * trace(M) ^ 2 = lambda1 * lambda2 - alpha (1ambda1 + lambda2) ^ 2
            float trace = (cv::trace(secondMomentMatrix))[0];
            float R = cv::determinant(secondMomentMatrix) - alpha * trace * trace;
            cornerResponse.at<float>(row, col) = R;
//            printRow("cornerResponse", cornerResponse);
        }
    }

    printRow("cornerResponse", cornerResponse);

    cv::Mat normalizedHarrisResponse;
    cv::normalize(cornerResponse, normalizedHarrisResponse, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    printRow("normalizedHarrisResponse", normalizedHarrisResponse);

    string responsePath = current_path + filename + "-harris-response.jpg";
    cv::imwrite(responsePath, normalizedHarrisResponse);

//    getCornerResponse( gradientX, gradientY, windowSize, sigmaGaussian, alpha, cornerResponse);
//
//    cv::normalize(cornerResponse, normalizedHarrisResponse, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//    cv::imwrite(responsePath, normalizedHarrisResponse);
//
//    printRow("gradientX", gradientX);
//    printRow("gradientY", gradientY);
//    printRow("cornerResponse", cornerResponse);
//    printRow("normalizedHarrisResponse", normalizedHarrisResponse);


    const int minDistance = 5;
    const double threshold = 500000000;
    static constexpr size_t SIFT_WINDOW_SIZE = 10;
    std::vector<std::pair<int, int>> cornerLocs;

    assert(cornerResponse.type() == CV_32F);
    cv::Mat localMaximaCorners = cv::Mat::zeros(cornerResponse.rows, cornerResponse.cols, cornerResponse.type());
    printRow("localMaximaCorners", localMaximaCorners);

    // collect local maxima
    for (int row = 0; row < cornerResponse.rows; row++) {
        for (int col = 0; col < cornerResponse.cols; col++) {
            float cornerValue = cornerResponse.at<float>(row, col);
            if (cornerValue >= threshold) {
                bool isLocalMaxima = true;

                for (int rowWeight = -minDistance; rowWeight < minDistance; rowWeight++) {
                    for (int colWeight = -minDistance; colWeight < minDistance; colWeight++) {
                        int compareRow = std::min(std::max(0, row + rowWeight), cornerResponse.rows - 1);
                        int compareCol = std::min(std::max(0, col + colWeight), cornerResponse.cols - 1);
                        if (row == compareRow && col == compareCol) continue;

                        if (cornerValue <= cornerResponse.at<float>(compareRow, compareCol)) {
                            isLocalMaxima = false;
                            break;
                        }
                    }
                    if (!isLocalMaxima) break;
                }

                if (isLocalMaxima) {
                    localMaximaCorners.at<float>(row, col) = cornerValue;
                    cornerLocs.push_back(std::make_pair(row, col));

                    // skip ahead in the row search
                    col += minDistance - 1;
                }
            }
        }
    }

    cv::Mat dottedImg;
    drawDots(localMaximaCorners, input, dottedImg);

    string localMaximaPath = current_path + filename + "-harris-localmaxima.jpg";
    cv::imwrite(localMaximaPath, dottedImg);
    printRow("dottedImg", dottedImg);

}

#include <vector>
#include <cmath>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

struct FeaturesContainer {
    cv::Mat gradientX, gradientY, cornerResponse, corners;
    std::vector<std::pair<int, int>> cornerLocs;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::DMatch> goodMatches;
    cv::Mat descriptors;
    cv::Mat input, drawnKeypoints;

    FeaturesContainer() {}
};

void computeScaleInvariant(FeaturesContainer& container, string current_path, string filename) {
    string imgPath = current_path + filename + ".jpg";

    cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    printRow("img", img);

    img.convertTo(container.input, CV_32F);
    printRow("input", container.input);

    int sobel_kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;

    cv::Sobel(container.input, container.gradientX, ddepth, 1, 0, sobel_kernel_size, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(container.input, container.gradientY, ddepth, 0, 1, sobel_kernel_size, scale, delta, cv::BORDER_DEFAULT);
    printRow("gradientX", container.gradientX);
    printRow("gradientY", container.gradientY);


    container.cornerResponse = cv::Mat::zeros(container.gradientX.rows, container.gradientX.cols, CV_32F);
    printRow("cornerResponse", container.cornerResponse);

    const size_t windowSize = 5;
    const double sigmaGaussian = 1.5;
    const float alpha = 0.04;
    cv::Mat kernelGaussian2D = cv::getGaussianKernel(windowSize, sigmaGaussian, container.gradientX.type());
    kernelGaussian2D = kernelGaussian2D * kernelGaussian2D.t(); // generate a 2d matrix with an outer product
    print("kernelGaussian1D", kernelGaussian2D);

    // compute a second moment matrix for each pixel, where t    static constexpr size_t SIFT_WINDOW_SIZE = 10;
    //he weights are the Gaussian kernel
    int windowHalf = windowSize / 2;
    for (int row = 0; row < container.gradientX.rows; row++) {
        for (int col = 0; col < container.gradientX.cols; col++) {
            cv::Mat secondMomentMatrix = cv::Mat::zeros(2, 2, CV_32F);

            for (int rowWeight = -windowHalf; rowWeight <= windowHalf; rowWeight++) {
                for (int colWeight = -windowHalf; colWeight <= windowHalf; colWeight++) {

                    float xGradientAt = container.gradientX.at<float>(std::min(std::max(0, row + rowWeight), container.gradientX.rows - 1),
                                                            std::min(std::max(0, col + colWeight), container.gradientX.cols - 1));

                    float yGradientAt = container.gradientY.at<float>(std::min(std::max(0, row + rowWeight), container.gradientY.rows - 1),
                                                            std::min(std::max(0, col + colWeight), container.gradientY.cols - 1));

                    // a gradient matrix
                    cv::Mat gradientMatrix = (cv::Mat_<float>(2, 2)  <<
                                                                     xGradientAt * xGradientAt,
                            xGradientAt * yGradientAt,
                            xGradientAt * yGradientAt,
                            yGradientAt * yGradientAt);

                    float weightXY = kernelGaussian2D.at<float>(rowWeight + windowHalf, colWeight + windowHalf);
                    secondMomentMatrix = secondMomentMatrix + weightXY * gradientMatrix;
                }
            }

            // compute Harris corner response
            // R = det(M) - alpha * trace(M) ^ 2 = lambda1 * lambda2 - alpha (1ambda1 + lambda2) ^ 2
            float trace = (cv::trace(secondMomentMatrix))[0];
            float R = cv::determinant(secondMomentMatrix) - alpha * trace * trace;
            container.cornerResponse.at<float>(row, col) = R;
        }
    }

    printRow("cornerResponse", container.cornerResponse);

    const int minDistance = 5;
    const double threshold = 500000000;    static constexpr size_t SIFT_WINDOW_SIZE = 10;

    assert(container.cornerResponse.type() == CV_32F);
    container.corners = cv::Mat::zeros(container.cornerResponse.rows, container.cornerResponse.cols, container.cornerResponse.type());

    // collect local maxima
    for (int row = 0; row < container.cornerResponse.rows; row++) {
        for (int col = 0; col < container.cornerResponse.cols; col++) {
            float cornerValue = container.cornerResponse.at<float>(row, col);
            if (cornerValue >= threshold) {
                bool isLocalMaxima = true;

                for (int rowWeight = -minDistance; rowWeight < minDistance; rowWeight++) {
                    for (int colWeight = -minDistance; colWeight < minDistance; colWeight++) {
                        int compareRow = std::min(std::max(0, row + rowWeight), container.cornerResponse.rows - 1);
                        int compareCol = std::min(std::max(0, col + colWeight), container.cornerResponse.cols - 1);
                        if (row == compareRow && col == compareCol) continue;

                        if (cornerValue <= container.cornerResponse.at<float>(compareRow, compareCol)) {
                            isLocalMaxima = false;
                            break;
                        }
                    }
                    if (!isLocalMaxima) break;
                }

                if (isLocalMaxima) {
                    container.corners.at<float>(row, col) = cornerValue;
                    container.cornerLocs.push_back(std::make_pair(row, col));

                    // skip ahead in the row search
                    col += minDistance - 1;
                }
            }
        }
    }

    // Make sure our input images are the right size, then resize the output to the correct size
    assert(container.gradientX.rows == container.gradientY.rows &&
    container.gradientX.cols == container.gradientY.cols &&
    container.gradientX.type() == container.gradientY.type() &&
    container.gradientX.type() == CV_32F);

    container.keypoints.clear();

    for (const auto& corner : container.cornerLocs) {
        float Ix = container.gradientX.at<float>(corner.first, corner.second);
        float Iy = container.gradientY.at<float>(corner.first, corner.second);

        static constexpr float PI = 3.1415921636;
        float angle = std::atan2(Iy, Ix) * 100.f / PI;

        static constexpr size_t SIFT_WINDOW_SIZE = 10;
        container.keypoints.emplace_back(corner.second, corner.first, SIFT_WINDOW_SIZE, angle, 0);
    }

//    container.drawnKeypoints = cv::Mat::zeros( container.input.size(), CV_8UC3 );
    container.input.convertTo(container.input, CV_8UC3);
    cv::drawKeypoints(container.input, container.keypoints, container.drawnKeypoints,
            cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string localMaximaPath = current_path + filename + "-drawnKeypoints.jpg";
    cv::imwrite(localMaximaPath, container.drawnKeypoints);
    printRow("drawnKeypoints", container.drawnKeypoints);


    auto featuresSIFT = cv::xfeatures2d::SIFT::create();
    featuresSIFT->compute(container.input, container.keypoints, container.descriptors);
}

int matchKNN(string suffix) {


    std::vector<FeaturesContainer> containers;
    containers.emplace_back();
    containers.emplace_back();

    computeScaleInvariant(containers[0], "../", suffix + "A");
    computeScaleInvariant(containers[1], "../", suffix + "B");

    containers[0].goodMatches.clear();
    containers[1].goodMatches.clear();

    // Use KNN to find 2 matches for each point so we can apply the ratio test from the original
    // SIFT paper (https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
    std::vector<std::vector<cv::DMatch>> rawMatches;

    auto matcher = cv::BFMatcher::create();
    matcher->knnMatch(containers[0].descriptors, containers[1].descriptors, rawMatches, 2);
    for (const auto& matchPair : rawMatches) {
        if (matchPair[0].distance < 0.75 * matchPair[1].distance) {
            containers[0].goodMatches.push_back(matchPair[0]);
        }
    }
    // Copy good matches from img1 to img2
    containers[1].goodMatches = containers[0].goodMatches;

    // Create image with lines drawn between matched points. As we iterate through each point, log
    // its info
    cv::Mat combinedSrc;
    cv::hconcat(containers[0].input, containers[1].input, combinedSrc);
    cv::cvtColor(combinedSrc, combinedSrc, cv::COLOR_GRAY2RGB);

    std::stringstream ss;
    ss << "\nMatches:";
    cv::RNG rng(12345);
    for (const auto& match : containers[0].goodMatches) {
        cv::KeyPoint k1 = containers[0].keypoints[match.queryIdx];
        cv::KeyPoint k2 = containers[1].keypoints[match.trainIdx];
        int xOffset = containers[0].input.cols;
        cv::line(combinedSrc,
                 k1.pt,
                 cv::Point2f(k2.pt.x + xOffset, k2.pt.y),
                 cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        ss << "\nqueryIdx=" << match.queryIdx << "; trainIdx=" << match.trainIdx
           << "; distance=" << match.distance;
    }

    std::cout << ss.str() << std::endl;
    std::cout << "Found good matches:" << containers[0].goodMatches.size() << std::endl;
    string combinedPath = "../" + suffix + "-combined.jpg";
    cv::imwrite(combinedPath, combinedSrc);
}

inline float euclidianDist(const cv::Point& p, const cv::Point& q) {
    cv::Point diff = p - q;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

static std::mt19937 rng;
static bool seeded = false;
void seed(std::shared_ptr<std::seed_seq> seq) {
    if (!seeded) {
        rng.seed(*seq);
        seeded = true;
    }
}

enum class TransformType { TRANSLATION = 1, SIMILARITY = 2, AFFINE = 3 };
std::tuple<cv::Mat, std::vector<int>, double> solve(const std::vector<cv::Point2f>& srcPts,
                                                            const std::vector<cv::Point2f>& destPts,
                                                            const TransformType whichTransform,
                                                            const int ransacReprojThresh,
                                                            const int maxIters,
                                                            const double minConsensusRatio) {
    // Source and training dataset sizes must be the same
    assert(srcPts.size() == destPts.size());
    const size_t MINIMUM_SET = static_cast<size_t>(whichTransform);

    // Create vector of indices to select sample set and test set
    const size_t numPts = srcPts.size();
    std::vector<int> indices(numPts);
    std::iota(indices.begin(), indices.end(), 0);

    // Mersenne twister engine
    // std::random_device rd;
    // rng.seed(rd());

    std::vector<int> consensusSet;
    double consensusRatio = 0;
    int iterations = 0;
    cv::Mat transform;

    while (consensusRatio < minConsensusRatio && iterations < maxIters) {
        // Random shuffle the indices
        std::shuffle(indices.begin(), indices.end(), rng);

        // Compute transform matrix based on which transform type we're using
        switch (whichTransform) {
            case TransformType::TRANSLATION: {
                cv::Point2f pt1(srcPts[indices[0]]), pt1Prime(destPts[indices[0]]);

                // clang-format off
                // Translation matrix is the same as similarity, but with no rotation
                transform = (cv::Mat_<float>(2, 3) << 1, 0, pt1Prime.x - pt1.x,
                        0, 1, pt1Prime.y - pt1.y);
                // clang-format on
                break;
            }
            case TransformType::SIMILARITY: {
                // The first two make up the sample set; the rest are used for testing
                cv::Point2f pt1(srcPts[indices[0]]), pt2(srcPts[indices[1]]),
                        pt1Prime(destPts[indices[0]]), pt2Prime(destPts[indices[1]]);
                // clang-format off
                // We have a system of equations
                //  [u'      [a  -b   c   [u
                //   v']  =   b   a   d] * v]
                cv::Mat A = (cv::Mat_<float>(4, 4) <<   pt1.x, -pt1.y, 1, 0,
                        pt1.y,  pt1.x, 0, 1,
                        pt2.x, -pt2.y, 1, 0,
                        pt2.y,  pt2.x, 0, 1);
                cv::Mat b = (cv::Mat_<float>(4, 1) <<   pt1Prime.x,
                        pt1Prime.y,
                        pt2Prime.x,
                        pt2Prime.y);
                // clang-format on
                // Solve system Ax = b
                cv::Mat x;
                cv::solve(A, b, x);

                // std::cout << "Matrix x = \n" << x << std::endl;
                // clang-format off
                transform = (cv::Mat_<float>(2, 3) << x.at<float>(0, 0), -x.at<float>(1, 0), x.at<float>(2, 0),
                        x.at<float>(1, 0), x.at<float>(0, 0), x.at<float>(3, 0));
                // clang-format on
                break;
            }
            case TransformType::AFFINE: {
                // Take the first three points as the sample set
                cv::Point2f pt1(srcPts[indices[0]]), pt2(srcPts[indices[1]]), pt3(srcPts[indices[2]]),
                        pt1Prime(destPts[indices[0]]), pt2Prime(destPts[indices[1]]),
                        pt3Prime(destPts[indices[2]]);

                // clang-format off
                // Treat the affine transform matrix as a 3x3 with the last row [0, 0, 1]
                // -> Pad the columns of the three input points with 1s to make 3x3 matrices
                //      P' = T*P
                // Where P is a 3x3 of source points, P' is a 3x3 of dest points, and T is the 3x3
                // transformation matrix. T can then be computed with
                //      T = P'*P^(-1)
                cv::Mat Pprime = (cv::Mat_<float>(3, 3) << pt1Prime.x, pt2Prime.x, pt3Prime.x,
                        pt1Prime.y, pt2Prime.y, pt3Prime.y,
                        1     ,     1     ,     1      );
                cv::Mat P = (cv::Mat_<float>(3, 3) << pt1.x, pt2.x, pt3.x,
                        pt1.y, pt2.y, pt3.y,
                        1  ,   1  ,   1   );
                //clang-format on
                transform = Pprime * P.inv();
                transform = transform.rowRange(0, 2);
            }
        }

        // Iterate over the rest of the points and find the consensus set
        std::vector<int> curConsensusSet;
        for (int idx = MINIMUM_SET; idx < numPts; idx++) {
            cv::Point2f pt(srcPts[indices[idx]]), ptPrime(destPts[indices[idx]]);
            cv::Mat testX = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1);
            cv::Mat testB = transform * testX;
            // std::cout << "result = \n" << testB << std::endl;
            // Compute euclidian distance between the transformed point and the actual point
            float dist =
                    euclidianDist(cv::Point2f(testB.at<float>(0, 0), testB.at<float>(1, 0)), ptPrime);
            // std::cout << "Distance = " << dist << std::endl;

            // If the distance is small enough, add it to the consensus set
            if (dist <= ransacReprojThresh) {
                curConsensusSet.push_back(idx);
            }
        }

        // If the consensus ratio for this set is greater than the current largest consensus ratio,
        // set the largest consensus ratio and set to this iteration's
        double curConsensusRatio = double(curConsensusSet.size()) / double(numPts);
        if (curConsensusRatio > consensusRatio) {
            consensusRatio = curConsensusRatio;
            consensusSet = curConsensusSet;
        }
        iterations++;
    }

//    auto flogger = spdlog::get(config::FILE_LOGGER);
//    flogger->info("RANSAC took {} iterations", iterations);

    return std::make_tuple(transform, consensusSet, consensusRatio);
}

int sampleRandomConsensus(string suffix) {
    std::vector<FeaturesContainer> containers;
    containers.emplace_back();
    containers.emplace_back();

    computeScaleInvariant(containers[0], "../", suffix + "A");
    computeScaleInvariant(containers[1], "../", suffix + "B");

    containers[0].goodMatches.clear();
    containers[1].goodMatches.clear();

    // Use KNN to find 2 matches for each point so we can apply the ratio test from the original
    // SIFT paper (https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
    std::vector<std::vector<cv::DMatch>> rawMatches;

    auto matcher = cv::BFMatcher::create();
    matcher->knnMatch(containers[0].descriptors, containers[1].descriptors, rawMatches, 2);
    for (const auto& matchPair : rawMatches) {
        if (matchPair[0].distance < 0.75 * matchPair[1].distance) {
            containers[0].goodMatches.push_back(matchPair[0]);
        }
    }
    // Copy good matches from img1 to img2
    containers[1].goodMatches = containers[0].goodMatches;

    // Create image with lines drawn between matched points. As we iterate through each point, log
    // its info
    cv::Mat combinedSrc;
    cv::hconcat(containers[0].input, containers[1].input, combinedSrc);
    cv::cvtColor(combinedSrc, combinedSrc, cv::COLOR_GRAY2RGB);

    std::stringstream ss;
    ss << "\nMatches:";
    cv::RNG rng(12345);
    for (const auto& match : containers[0].goodMatches) {
        cv::KeyPoint k1 = containers[0].keypoints[match.queryIdx];
        cv::KeyPoint k2 = containers[1].keypoints[match.trainIdx];
        int xOffset = containers[0].input.cols;
        cv::line(combinedSrc,
                 k1.pt,
                 cv::Point2f(k2.pt.x + xOffset, k2.pt.y),
                 cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        ss << "\nqueryIdx=" << match.queryIdx << "; trainIdx=" << match.trainIdx
           << "; distance=" << match.distance;
    }

    std::cout << ss.str() << std::endl;
    std::cout << "Found good matches:" << containers[0].goodMatches.size() << std::endl;
    string combinedPath = "../" + suffix + "-combined.jpg";
    cv::imwrite(combinedPath, combinedSrc);

    // collect samples with (1 - p)^e
    std::vector<cv::Point2f> pointsTranslateA, pointsTranslateB;
    for (const auto& match: containers[0].goodMatches) {
        pointsTranslateA.emplace_back(containers[0].keypoints[match.queryIdx].pt);
        pointsTranslateB.emplace_back(containers[1].keypoints[match.trainIdx].pt);
    }

    int reprojection_threshold = 10;
    int max_iterations = 2000;
    float minimumConsensusRatio = 0.2;

    TransformType transformType = TransformType::TRANSLATION;
    cv::Mat transformTranslate;
    std::vector<int> inlierSet;
    double outlierRatio;
    std::tie(transformTranslate, inlierSet, outlierRatio) = solve(pointsTranslateA, pointsTranslateB,
            transformType, reprojection_threshold, max_iterations, minimumConsensusRatio);

    cv::Mat combinedImage;
    cv::hconcat(containers[0].input, containers[1].input, combinedImage);
    printRow("input0", containers[0].input);
    printRow("input1", containers[1].input);
    printRow("combinedImage", combinedImage);

    cv::cvtColor(combinedImage, combinedImage, cv::COLOR_GRAY2RGB);
    printRow("COLOR_GRAY2RGB", combinedImage);


    for (const int& index : inlierSet) {
        cv::line(combinedImage, pointsTranslateA[index],
                cv::Point2f(pointsTranslateB[index].x + containers[0].input.cols, pointsTranslateB[index].y),
                cv::Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0, 255)));
    }

    string inlierPath = "../" + suffix + "-inliers.jpg";
    cv::imwrite(inlierPath, combinedImage);
}

int transformSimilarityAffineByInlierSamples(string suffix) {
    std::vector<FeaturesContainer> containers;
    containers.emplace_back();
    containers.emplace_back();

    computeScaleInvariant(containers[0], "../", suffix + "A");
    computeScaleInvariant(containers[1], "../", suffix + "B");

    containers[0].goodMatches.clear();
    containers[1].goodMatches.clear();

    // Use KNN to find 2 matches for each point so we can apply the ratio test from the original
    // SIFT paper (https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
    std::vector<std::vector<cv::DMatch>> rawMatches;

    auto matcher = cv::BFMatcher::create();
    matcher->knnMatch(containers[0].descriptors, containers[1].descriptors, rawMatches, 2);
    for (const auto& matchPair : rawMatches) {
        if (matchPair[0].distance < 0.75 * matchPair[1].distance) {
            containers[0].goodMatches.push_back(matchPair[0]);
        }
    }
    // Copy good matches from img1 to img2
    containers[1].goodMatches = containers[0].goodMatches;

    // Create image with lines drawn between matched points. As we iterate through each point, log
    // its info
    cv::Mat combinedSrc;
    cv::hconcat(containers[0].input, containers[1].input, combinedSrc);
    cv::cvtColor(combinedSrc, combinedSrc, cv::COLOR_GRAY2RGB);

    std::stringstream ss;
    ss << "\nMatches:";
    cv::RNG rng(12345);
    for (const auto& match : containers[0].goodMatches) {
        cv::KeyPoint k1 = containers[0].keypoints[match.queryIdx];
        cv::KeyPoint k2 = containers[1].keypoints[match.trainIdx];
        int xOffset = containers[0].input.cols;
        cv::line(combinedSrc,
                 k1.pt,
                 cv::Point2f(k2.pt.x + xOffset, k2.pt.y),
                 cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        ss << "\nqueryIdx=" << match.queryIdx << "; trainIdx=" << match.trainIdx
           << "; distance=" << match.distance;
    }

    std::cout << ss.str() << std::endl;
    std::cout << "Found good matches:" << containers[0].goodMatches.size() << std::endl;
    string combinedPath = "../" + suffix + "-combined.jpg";
    cv::imwrite(combinedPath, combinedSrc);

    // collect samples with (1 - p)^e
    std::vector<cv::Point2f> pointsTranslateA, pointsTranslateB;
    for (const auto& match: containers[0].goodMatches) {
        pointsTranslateA.emplace_back(containers[0].keypoints[match.queryIdx].pt);
        pointsTranslateB.emplace_back(containers[1].keypoints[match.trainIdx].pt);
    }

    int reprojection_threshold = 6;
    int max_iterations = 2000;
    float minimumConsensusRatio = 0.6;

    TransformType transformType = TransformType::SIMILARITY;
    cv::Mat transformAffine;
    std::vector<int> inlierSet;
    double outlierRatio;
    std::tie(transformAffine, inlierSet, outlierRatio) = solve(pointsTranslateA,
            pointsTranslateB,
            transformType,
            reprojection_threshold, max_iterations, minimumConsensusRatio);

    cv::Mat transformAffineInverted;
    cv::invertAffineTransform(transformAffine, transformAffineInverted);
    print("transformAffine", transformAffine);
    print("transformAffineInverted", transformAffineInverted);

    cv::Mat simA = containers[0].input.clone();
    cv::Mat simB = containers[1].input.clone();

    cv::Mat reversedByAffine = cv::Mat::zeros(simB.rows, simB.cols, simB.type());
    cv::warpAffine(simB, reversedByAffine, transformAffineInverted, reversedByAffine.size());
    printRow("reversedByAffine", reversedByAffine);

    cv::Mat blendImage = simA * 0.5 + reversedByAffine * 0.5;
    printRow("blendImage", blendImage);

    string blendedPath = "../" + suffix + "-blended.jpg";
    cv::imwrite(blendedPath, blendImage);
}

int transformAffineByInlierSamples(string suffix) {
    std::vector<FeaturesContainer> containers;
    containers.emplace_back();
    containers.emplace_back();

    computeScaleInvariant(containers[0], "../", suffix + "A");
    computeScaleInvariant(containers[1], "../", suffix + "B");

    containers[0].goodMatches.clear();
    containers[1].goodMatches.clear();

    // Use KNN to find 2 matches for each point so we can apply the ratio test from the original
    // SIFT paper (https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
    std::vector<std::vector<cv::DMatch>> rawMatches;

    auto matcher = cv::BFMatcher::create();
    matcher->knnMatch(containers[0].descriptors, containers[1].descriptors, rawMatches, 2);
    for (const auto& matchPair : rawMatches) {
        if (matchPair[0].distance < 0.75 * matchPair[1].distance) {
            containers[0].goodMatches.push_back(matchPair[0]);
        }
    }
    // Copy good matches from img1 to img2
    containers[1].goodMatches = containers[0].goodMatches;

    // Create image with lines drawn between matched points. As we iterate through each point, log
    // its info
    cv::Mat combinedSrc;
    cv::hconcat(containers[0].input, containers[1].input, combinedSrc);
    cv::cvtColor(combinedSrc, combinedSrc, cv::COLOR_GRAY2RGB);

    std::stringstream ss;
    ss << "\nMatches:";
    cv::RNG rng(12345);
    for (const auto& match : containers[0].goodMatches) {
        cv::KeyPoint k1 = containers[0].keypoints[match.queryIdx];
        cv::KeyPoint k2 = containers[1].keypoints[match.trainIdx];
        int xOffset = containers[0].input.cols;
        cv::line(combinedSrc,
                 k1.pt,
                 cv::Point2f(k2.pt.x + xOffset, k2.pt.y),
                 cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        ss << "\nqueryIdx=" << match.queryIdx << "; trainIdx=" << match.trainIdx
           << "; distance=" << match.distance;
    }

    std::cout << ss.str() << std::endl;
    std::cout << "Found good matches:" << containers[0].goodMatches.size() << std::endl;
    string combinedPath = "../" + suffix + "-combined.jpg";
    cv::imwrite(combinedPath, combinedSrc);

    // collect samples with (1 - p)^e
    std::vector<cv::Point2f> pointsTranslateA, pointsTranslateB;
    for (const auto& match: containers[0].goodMatches) {
        pointsTranslateA.emplace_back(containers[0].keypoints[match.queryIdx].pt);
        pointsTranslateB.emplace_back(containers[1].keypoints[match.trainIdx].pt);
    }

    int reprojection_threshold = 6;
    int max_iterations = 2000;
    float minimumConsensusRatio = 0.6;

    TransformType transformType = TransformType::AFFINE;
    cv::Mat transformAffine;
    std::vector<int> inlierSet;
    double outlierRatio;
    std::tie(transformAffine, inlierSet, outlierRatio) = solve(pointsTranslateA,
                                                               pointsTranslateB,
                                                               transformType,
                                                               reprojection_threshold, max_iterations, minimumConsensusRatio);

    cv::Mat transformAffineInverted;
    cv::invertAffineTransform(transformAffine, transformAffineInverted);
    print("transformAffine", transformAffine);
    print("transformAffineInverted", transformAffineInverted);

    cv::Mat simA = containers[0].input.clone();
    cv::Mat simB = containers[1].input.clone();

    cv::Mat reversedByAffine = cv::Mat::zeros(simB.rows, simB.cols, simB.type());
    cv::warpAffine(simB, reversedByAffine, transformAffineInverted, reversedByAffine.size());
    printRow("reversedByAffine", reversedByAffine);

    cv::Mat blendImage = simA * 0.5 + reversedByAffine * 0.5;
    printRow("blendImage", blendImage);

    string affinePath = "../" + suffix + "-affine.jpg";
    cv::imwrite(affinePath, blendImage);
}

int main() {
//    fundamental();

//    computeHarrisResponse("../", "simA");
//    computeHarrisResponse("../", "simB");
//    computeHarrisResponse("../", "transA");
//    computeHarrisResponse("../", "transB");

//    matchKNN("sim");
//    matchKNN("trans");

//    sampleRandomConsensus("sim");
//    transformSimilarityAffineByInlierSamples("trans");
    transformAffineByInlierSamples("trans");

}
