
*  **Multiple Regression from Simple Univariate Regression**
     * Inner product notation and relationship to multiple regression.
     * Orthogonal predictors and their effect on parameter estimates
     * Orthogonalization procedures: adjustment for x0, steps for simple regression and multiple regression interpretations.
    * Gram-Schmidt procedure: Algorithm for orthogonalization.

*   **Multiple Outputs:**
    *   Multiple linear models: Y = XB + E.
    *   Multivariate loss: tr[(Y - XB)ᵀ(Y - XB)].
    *   Least squares solution: B = (XᵀX)⁻¹XᵀY.
    *   Multivariate weighted criterion: considering correlation of errors (Σ) and its result in independent regressions.

*   **Subset Selection:**
    *   Motivations: Improve prediction accuracy by reducing variance, and enhance model interpretation.
    *   **3.3.1 Best-Subset Selection:**
        *   Finding best model for each subset size k.
        *   Leaps and bounds procedure
        *   Tradeoff between bias and variance for model selection using criteria like Cross-Validation and AIC.
    *   **3.3.2 Forward- and Backward-Stepwise Selection:**
        *   Sequential addition (forward) or deletion (backward) of predictors.
        *   Computational advantages for large p.
        *   Variance-bias tradeoff and potential sub-optimality
    *   **3.3.3 Forward-Stagewise Regression:**
        *   Constrained approach, iteratively adding predictor that correlates most with residual with coefficients gradually adjusted.
         *  "Slow fitting" and potential benefits in high-dimensional problems.
    *   **3.3.4 Prostate Cancer Data Example (Continued):**
        *   Comparing coefficients across different selection and shrinkage methods with cross-validation error estimates.

*   **Shrinkage Methods:**
    *   Overview: Addressing high variability by continuous shrinkage methods
    *   **3.4.1 Ridge Regression:**
        *   Penalized residual sum of squares: minimizing ||y - Xβ||² + λ||β||².
        *   Size constraint interpretation: min ||y - Xβ||² subject to  ||β||² ≤ t.
        *   Handling correlated predictors by penalization.
        *   Ridge solution: βridge = (XᵀX + λI)⁻¹Xᵀy.
        *   Connection to Bayesian methods: posterior mode from Gaussian priors.
        *   Singular Value Decomposition (SVD) interpretation: shrinking components based on singular values and principal components.
        *   Profiles of ridge coefficients vs. effective degrees of freedom.
    *   **3.4.2 The Lasso:**
        *   L1 penalty: minimizing ||y - Xβ||² subject to ||β||₁ ≤ t
        *   Continuous subset selection via coefficient shrinkage to zero.
        *   Basis pursuit analogy and Lagrangian form: minimizing ||y - Xβ||² + λ||β||₁.
        *   Contrasting L1 vs L2 penalties and their impact.
    *   **3.4.3 Discussion: Subset Selection, Ridge Regression, and the Lasso**
        *   Comparison across methods in the orthogonal case and by means of the explicit solution.
        *    Different types of shrinkage: proportional, soft, and hard.
        *    Visualizations with elliptic and diamond constraint regions, their implications for model complexity.
    *   **3.4.4 Least Angle Regression:**
        *   "Democratic" version of forward stepwise with partial predictor inclusion and correlated variable tracking.
        *   Algorithm details: iterative process of identifying, fitting, and moving coefficients.
        *   Relation to Lasso: derivation of lasso solution paths from modified LARS.

*  **Methods Using Derived Input Directions:**
    *  Overview: generating reduced number of linear combinations of inputs as alternative models
    * **3.5.1 Principal Components Regression (PCR):**
        *   Using principal components as input variables for regression and truncated regression.
        *   Regression on derived orthogonal components; connection to ridge regression.
    * **3.5.2 Partial Least Squares (PLS):**
        *   Constructing inputs based on response and predictors.
        *  Using y for weighting inputs and iterative orthogonalization.
        *  Non-linear behavior with y, yet a similar behavior to PCR and Ridge.

*   **A Comparison of the Selection and Shrinkage Methods**
    *   Comparison of methods using two correlated predictors by plotting coefficient profiles and discussing different shrinkage and selection behaviours.
        * The shrinkage process and their characteristics (discrete or continuous).

*   **Multiple Outcome Shrinkage and Selection:**
    *   Univariate vs. Multivariate shrinkage/selection.
    *   Canonical Correlation Analysis (CCA): maximizing correlation between linear combinations of inputs and responses.
    *   Reduced-Rank Regression: explicit model with pooled information.
    *   Shrinking canonical variates.

*   **Lasso and Related Path Algorithms**
    *   **3.8.1 Incremental forward Stagewise regression**
        * New algorithm based on forward stagewise regression
        * Step size and the relation between LASSO
    *  **3.8.2 Piecewise-Linear path algorithm**
        *  Exploring piecewise linear nature of solution paths from regularized problems by using convex loss and penalty.
    * **3.8.3 The Dantzig Selector:**
        *   Alternative to Lasso with different objective to reduce max inner product rather than sum of errors.
        *   Connection to Linear programming.
        *   Practical Issues.
    *   **3.8.4 The Grouped Lasso:**
        *   Extending lasso to group-structured predictors
    *   **3.8.5 Further Properties of the Lasso**
        *   Theoretical guarantees of lasso model recovery and consistency.

*   **Pathwise Coordinate Optimization**
        *   Alternative for LARS based on coordinate descent.
        *   Cyclical optimization over parameters.

*   **Computational Considerations:**
    *   Efficiency of Cholesky/QR for least squares.
    *   LARS complexity as a comparable algorithm.