/**
 * StructureProjectionFactor.h
 * 
 * @author Sean Bowman
 */

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/JacobianFactor.h>

#include <boost/optional.hpp>
#include <boost/make_shared.hpp>

#include "geometry.h"

class StructureProjectionFactorTester;

namespace semslam {

using geometry::ObjectModelBasis;

/**
 * Factor representing the cost function of an object defined by a set of deformable keypoints and their
 * projection onto a calibrated camera
 */
class StructureProjectionFactor : public gtsam::NonlinearFactor {
public:
    
    typedef boost::shared_ptr<StructureProjectionFactor> shared_ptr;
    typedef shared_ptr Ptr;

    /**
     * Parameters:
     *   normalized_measurements - 3xN matrix of normalized image coordinates of keypoint observations
     *   object_key - the key corresponding to the object's pose in the factor graph
     *   landmark_keys - keys of each keypoint's 3d position in the factor graph
     *   coefficient_key - key of the structure coefficient vector within the factor graph
     *   model           - object model
     *   weights         - Nx1 vector of observation weights
     *   lambda          - regularization coefficient on ||c||
     */
    StructureProjectionFactor(const Eigen::MatrixXd& normalized_measurements,
                    gtsam::Key object_key,
                    const std::vector<gtsam::Key>& landmark_keys,
                    const gtsam::Key& coefficient_key,
                    const ObjectModelBasis& model,
                    const Eigen::VectorXd& weights,
                    double lambda=1.0);

    void setWeights(const Eigen::VectorXd& weights) { weights_ = weights; }
    
    /**
     * Returns the unweighted residual vector r
     * r = [e^T c^T]^T
     * where e_i = w_i*z_i - R*s_i - t
     */
    gtsam::Vector unweightedError(const gtsam::Values& values,
                                  boost::optional< std::vector<gtsam::Matrix>& > = boost::none) const;

    /**
     * Applies weights (both the actual "weights" and the structure regularizing lambda) 
     * to an unweighted residual vector
     */
    gtsam::Vector weightedError(const gtsam::Values& values) const;

    /**
     * Returns the *total* error = 0.5*r^T*r
     * where r is the weighted residual vector
     */
    double error(const gtsam::Values& values) const;

    /**
     * Linearizes the factor at the current estimate, returning a factor corresponding to 
     * the resulting linear cost function
     */
    boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& values) const;

    /**
     * Dimension of the residual vector
     */
    size_t dim() const {
        return 3*m_ + k_;
    }


                                                                                                 
private:

    Eigen::MatrixXd measurements_;

    gtsam::Key object_key_;
    std::vector<gtsam::Key> landmark_keys_; 
    gtsam::Key coefficient_key_;

    ObjectModelBasis model_;

    double lambda_; // regularization term coefficient

    size_t m_, k_;

    Eigen::VectorXd weights_;

    std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> Bi_;

    Eigen::MatrixXd Dobject(const gtsam::Values& values) const;
    Eigen::MatrixXd Dcoefficients(const gtsam::Values& values) const;
    Eigen::VectorXd Dlandmark(const gtsam::Values& values, size_t landmark_index) const;

    Eigen::MatrixXd structure(const gtsam::Values& values) const;

    
    void weightError(gtsam::Vector& e) const;

    void weightJacobians(std::vector<gtsam::Matrix>& vec) const;

    void weightJacobian(gtsam::Matrix& H) const;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

} // namespace semslam
