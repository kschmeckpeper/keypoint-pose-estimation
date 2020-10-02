/**
 * geometry.h
 * 
 * Contains functions for optimizing object structure
 * 
 * @author Sean Bowman
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <limits>
#include <fstream>

namespace geometry { 

struct StructureResult {
    // Eigen::Matrix<double, 3, Eigen::Dynamic> S;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    Eigen::VectorXd C;

    Eigen::VectorXd Z;
    Eigen::VectorXd Z_covariance;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ObjectModelBasis {
    // mean shape
    Eigen::MatrixXd mu;

    // deformable basis components
    Eigen::MatrixXd pc;
};

ObjectModelBasis readModelFile(std::string file_name);

Eigen::MatrixXd centralize(const Eigen::MatrixXd& M);

Eigen::MatrixXd reshapeS_b2v(const Eigen::MatrixXd& S);

Eigen::MatrixXd composeShape(const Eigen::MatrixXd& B, const Eigen::VectorXd& C);

Eigen::VectorXd sample_stddev(const Eigen::MatrixXd& data);

/**
 * Optimizes an objects structure coefficients c (S = B0 + sum_i c_i*B_i) and its pose in SO(3)
 * given a set of keypoint observations normalized_coords and their corresponding weights.
 * 
 * Does so with a Levenberg-Marquardt optimization over all variables Z, t, R, C
 *
 * Inputs:
 *  normalized_coords  - a 3xN matrix of normalized image cordinates of keypoint observations
 *  model              - the object class's deformable model
 *  weights            - a Nx1 vector of observation weights
 *  compute_covariance - if true, in addition to computing estimates of the depth of each observed keypoint,
 *                       this function will also compute the covariance of that estimate
 */
StructureResult optimizeStructureFromProjection(const Eigen::MatrixXd& normalized_coords,
                                                ObjectModelBasis model,
                                                Eigen::VectorXd weights,
                                                bool compute_covariance=false);

/**
 * Optimizes an objects structure coefficients c (S = B0 + sum_i c_i*B_i) and its pose in SO(3)
 * given a set of keypoint observations normalized_coords an their corresponding weights.
 * 
 * Does so by performing coordinate descent over Z, t, R, C.
 * 
 * Inputs:
 *  normalized_coords  - a 3xN matrix of normalized image cordinates of keypoint observations
 *  model              - the object class's deformable model
 *  weights            - a Nx1 vector of observation weights
 */
StructureResult optimizeStructureFromProjectionCoordinateDescent(const Eigen::MatrixXd& normalized_coords,
                                                const ObjectModelBasis& model,
                                                const Eigen::VectorXd& weights); 

// Returns +1 if val > 0, 0 if val == 0, -1 if val < 0.
template <typename T>
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


Eigen::Matrix3d skewsymm(const Eigen::Vector3d& x) {
    Eigen::Matrix3d S;
    S <<   0,  -x(2), x(1),
          x(2),  0,  -x(0),
         -x(1), x(0),  0;
    return S;
}

} // namespace geometry
