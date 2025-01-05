*   **12.1 Introduction:**
    *   Limitations of Linear Decision Boundaries:  The restriction of linearity and the need for generalizations to handle overlapping classes.
    *   Support Vector Machines (SVMs):  Linear separation in high-dimensional feature spaces; resulting in non-linear decision boundaries in input feature spaces.
    *   Generalized LDA: Introduction of flexible and penalized discriminant analysis to address the limitations of linear discriminant analysis.
    *  Motivation of SVM and its link to other non-linear methods.

*   **12.2 The Support Vector Classifier:**
    *   Review of optimal separating hyperplane:  Separable classes, maximal margin, and its properties (Section 4.5.2).
        *  The hyperplane is defined by points whose distance to the decision boundary are maximal, and points on the margins.
    *   Nonseparable Cases: Introduction of slack variables ξ to allow points to fall on the "wrong side" of the margin, and two different ways to do so.
    *   Rephrasing of the optimization problem for nonseparable case:
        *  Minimizing a norm of coefficients plus an upper bound on the training misclassifications with a cost parameter C.
    *   Convex Optimization: A convex optimization problem (quadratic criterion with linear constraints).
    *   Lagrange multipliers for the primal, and the Wolfe dual objective.
        * Karush-Kuhn-Tucker Conditions for identifying unique solutions.

*   **12.2.1 Computing the Support Vector Classifier:**
    *   Rephrasing of the criterion, and the matrix notation for the dual and primal objective functions.
    *   Computation of the support vectors, parameters and the role of the parameter C in the solutions.
    *   Quadratic Programming: Convex quadratic programming approach to solve the problem using Lagrange multipliers.
    *   Support Vectors:  Observations that define the decision boundary.
    *   Sensitivity to C:  Relationship between the value of C and the sensitivity to points near and far from the decision boundary.
    *   Cross Validation: Estimation of C using cross-validation, and relationship between the number of support vectors and leave-one-out cross-validation error.

*   **12.3 Support Vector Machines and Kernels:**
    *   Concept: Generalization to nonlinear boundaries through linear separation in an enlarged feature space.
        *   Using kernels to avoid explicit transformations in high-dimensional feature space, using inner products.
        * Polynomial, radial, and neural network kernels as examples.
    *   Kernel definition: Symmetric, positive (semi-) definite functions that compute inner products in a higher dimensional space without explicitly transforming the input vector.
   *   Relationship between SVM and Basis functions, and their role in the model representation.

*   **12.3.1 Computing the SVM for Classification:**
    *   Kernel Functions: Use of kernels to express the transformation without actually computing the expanded features.
        * Explicit representation of the support vector classifier using kernels.
    *   SVD Interpretation: using SVD and its properties to analyze support vectors
*   **12.3.2 Mixture Example (Continued):**
    *  Performance analysis using nonlinear support vector machines with polynomial and radial kernels on the mixture data.
        *   Use of different kernel parameters and test error estimates.
*   **12.3.3 Function Estimation and Reproducing Kernels:**
    *   RKHS (Reproducing Kernel Hilbert Spaces): The connection of support vector classifiers to function fitting and reproducing kernel Hilbert spaces.
        *  Reproducing property of kernels and how the kernel can be used to evaluate function values at specific points.
     * The SVMs viewed as a penalized regression approach in a RKHS.
* **12.3.4 SVMs and the Curse of Dimensionality**
    *   Using kernel methods for large dimensional spaces by means of linear projections on different feature spaces and regularization.
    *   Use of SVM in a simulation setting and testing polynomial kernels as well as linear models.
* **12.3.5 A Path Algorithm for the SVM Classifier**
   *   Use of piecewise linear coefficient paths, and a connection of the points on the margin and the Lagrangian multipliers.
   *   Efficient techniques to solve the optimization with different tuning parameter values.
*   **12.3.6 Support Vector Machines for Regression:**
    *  Adapting SVMs to regression problems by using an epsilon-insensitive loss function and defining a penalized optimization problem with a constraint and a penalty parameter.
        *   Explicit connection between the Support Vector Regression with the Kernel methods via penalized regression and reproducing kernels.
   * Use of quadratic loss function to derive the weights.

*   **12.4 Generalizing Linear Discriminant Analysis:**
    *   Limitations of LDA:  Linear boundaries, single prototype per class, need for adaptations with flexible discriminant analysis.
    *   Advantages of LDA: Simplicity, interpretability, and performance in many situations
    *   Flexible Discriminant Analysis (FDA): Extension of LDA via linear regression of a transformed class indicator matrix.
    *  Penalized discriminant analysis: extension of FDA, incorporating smoothing and dimensionality reduction.
    *   Mixture Discriminant Analysis (MDA):  Use of Gaussian mixture models to model class densities and address the inadequacies of single prototype per class representation.

* **12.5 Flexible Discriminant Analysis:**
    *    Optimal scoring for LDA, which is based on the joint probability.
    *   Extension to flexible discriminant analysis via regression on the indicator response matrix, by selecting optimal scores.
    *  The use of a more general optimization criteria based on scores, and the link between penalized canonical correlations and discriminant analysis.
* **12.5.1 Computing the FDA Estimates:**
    *    Computations based on generalized eigenvectors.
    *  Relation between FDA and linear regression, which allows using it as a regression tool.

*  **12.6 Penalized Discriminant Analysis**
    *   Use of a basis expansion of a linear transformation that relates inputs and responses
        *   Implementation of smoothness constraints using a regularized Mahalanobis Distance.

* **12.7 Mixture Discriminant Analysis**
   *  Extending LDA to model each class via a Gaussian Mixture Model.
        * EM algorithm for estimation of parameters.
    *   Hierarchical modeling using weighted least squares, and rank constraints in multi-class scenarios.
  
*  **12.7.1 Example: Waveform Data:**
    *   Performance comparison of FDA, MDA and PDA on simulated data using a 3-class wave form problem.
        *  Better performance of MDA in this specific dataset.
       *  Use of simulated data to better understand the relationship between parameters.

