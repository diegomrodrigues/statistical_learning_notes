
*   **4.2 Linear Regression of an Indicator Matrix:**
    *   Indicator Coding: Representing classes by indicator vectors for regression.
    *   Linear regression: Fitting each output as linear function of inputs.
    *   Decision Rule: Classify to the largest fitted vector value.
    *   Statistical interpretation: view regression as an estimate of conditional expectation.
    *   Shortcomings: masking of classes when K is large; not constrained in [0,1].
    *  Simplified Viewpoint: targets from identity matrix, fitting by least squares.
    *   Connection between Sum-of-squares and indicator matrix regression.
    *  Consequence: components are separable, each with an individual linear model.
   
*   **4.3 Linear Discriminant Analysis (LDA):**
    *   Decision Theory: Basing decisions on posterior class probabilities Pr(G|X=x).
    *   Class-Conditional Densities: Models for densities: Gaussians, Flexible mixtures of Gaussians, Nonparametric estimates, Naive Bayes (conditional independence)
        * Modeling class densities as multivariate Gaussians with common covariance matrices.
    *   LDA Decision Boundary: Log ratio yields a linear decision boundary by means of equal covariance assumption.
    *   Linear Discriminant Functions.
    *   Estimating LDA parameters with training data: class proportions, means, pooled covariances.
    *   Connection to Least Squares: Proportionality of the coefficient vector of the decision rule to the coefficients of regression on a coded response.
        *   Cut point or intercept is different and does use Gaussian parameter.
    *   Masking Problem: Limitations of linear regression in multiclass problems; use LDA to overcome this issue.
    *   QDA: Quadratic discriminant analysis with different class covariance matrices, the resulting decision boundary is a quadratic equation.
        * Computational considerations with SVD of covariances.
*   **4.3.1 Regularized Discriminant Analysis:**
    *   Shrinking QDA to LDA with shrinkage covariance matrices and their practical implementation for validation purposes.
    *   Shrinking covariances toward a scalar (isotropic) covariance.
*   **4.3.2 Computations for LDA:**
    *   Eigen-decomposition of covariance matrices for efficient computations of decision boundaries.
    *   Sphering data with common covariance and closest class centroid strategy.
*  **4.3.3 Reduced-Rank Linear Discriminant Analysis:**
    *   Dimensionality reduction of centroids and projections onto informative subspaces using Fisher Criterion.
    *   Discriminant Coordinates (canonical variates), variance maximization and relation with SVD.
    *  Optimization problem with the Rayleigh quotient and the generalization of SVD.
        *  Maximizing between-class variance with minimum within-class variance.

*   **4.4 Logistic Regression:**
    *   Modeling posterior class probabilities with linear functions, enforcing values in [0,1] and sum to 1.
    *   Logit transformation: log-odds or logits to linearize the model.
    *   Multinomial Logit: the use of a reference class to simplify the models.
    *   Relation with Generalized linear model via a linear function.
    *  **4.4.1 Fitting Logistic Regression Models:**
        *   Maximizing the conditional likelihood.
        *   Log-Likelihood equation with two-class case and 0/1 response.
        *   Score equations and the use of the Newton-Raphson algorithm.
        *   Derivation of the Hessian matrix and the update step with the use of matrix notation.
        *   Iteratively reweighted least squares algorithm and their relation with logistic regression
        *   Use of coordinate-descent method to solve in large scale and R's package glmnet
    *   **4.4.2 Example: South African Heart Disease:**
        *   Example of fitting a logistic regression to a binary data, Z-scores for significance and possible interactions between variables.
        *   Model selection via stepwise regression by dropping least significant variables using residual deviance.
        *   Interpretation of coefficients with exponential odds and confidence intervals.
   * **4.4.3 Quadratic Approximations and Inference**
        *   Exploiting the relation between logistic regression and weighted least squares.
        *   Pearson chi-square statistic, asymptotic likelihood theory, central limit theorem and shortcuts based on maximum likelihood.
    *  **4.4.4 L1 Regularized Logistic Regression**
       *   Use of lasso penalty in logistic regression for variable selection and shrinkage, and a modification of the logistic regression optimization problem.
        * Interpretation of the score equations and relations with generalized lasso.
        * Path algorithm limitations.
*   **4.4.5 Logistic Regression or LDA?**
    *   Comparison between LDA and logistic regression: differences in assumption and fitting approach despite a shared mathematical form.
    *    LDA estimates with Gaussian assumptions, whereas logistic regression leaves Pr(X) unspecified.
    *    Use of marginal likelihood to provide more parameters and robustness, and discussion in context of outliers.
        *   Use of marginal likelihood as a regularizer and discussion of degeneracy with perfect separation.
        * Use of logistic regression and LDA in presence of qualitative variables, where logistic regression is more robust.

*  **4.5 Separating Hyperplanes:**
    *   Perceptrons as basis for neural networks
        *   Affine sets and the use of signed distances to a hyperplane.
    * **4.5.1 Rosenblatt's Perceptron Learning Algorithm:**
        *   Iterative strategy for finding a separating hyperplane by gradient descent using stochastic approaches.
        *   Convergence issues and a list of problems arising from the algorithm.
    *   **4.5.2 Optimal Separating Hyperplanes:**
        *   Optimization problem of maximizing the margin between classes using the maximum signed distance strategy.
        *    Introducing the Wolfe dual for optimization.
        *     Finding a solution as a linear combination of the support points and a classification rule for new observations
         * Support vectors define the optimal separation.
        *   Limitations with data that are not separable.
