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
