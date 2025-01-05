*   **15.1 Introduction:**
    *   Bagging as a Variance Reduction Tool: Review of bagging and the averaging of noisy but unbiased models.
    *   Trees as Candidates for Bagging: Their capacity to capture complex interactions, and high bias.
    *   Boosting and Bagging comparison: Highlighting the differences in terms of parameter estimation.
    *   Random Forests: A substantial modification of bagging that builds de-correlated trees, and their high performance when compared to boosting, that are simpler to train and tune.
    *   Popularity:  Wide use and implementation of random forests in many packages.

*   **15.2 Definition of Random Forests:**
    *   Averaging Unbiased Models: Reduction of the variance by averaging approximately unbiased models.
    *   Bootstrapped Trees: Building tree classifiers using random bootstrap sampling.
    *   Randomization: Introducing randomness into the tree-building process by splitting a tree on a selected subset of input features.
       *   Selection of m features to consider at each split, for p input variables in the whole dataset.
   *   Regression vs Classification: Different methods for predictions (averaging in regression, majority vote in classification).

*   **15.3 Details of Random Forests:**
    *   Tree Building Algorithm: A recursive tree building algorithm and the use of stochastic process in the splitting criteria.
        * Random selection of input variables ( m < p).
    *    Stopping rule: stopping the tree recursion with a minimum node size nmin.
   * Hyperparameters: m for selecting variables for the splits and min node size.
     *Recommendations for classification and regression with use of sqrt(p) and p/3.
    *   Variance-Reduction: Improvement in the variance reduction, by the reduction of correlation between the trees.
    *   **15.3.1 Out-of-Bag Samples:**
        *   OOB Samples: Samples not used for training a specific tree.
        *  OOB Error Estimation: Use of OOB samples for accurate model error estimation which is similar to cross validation.
    *  **15.3.2 Variable Importance:**
        *   Gini importance calculation and interpretations of variables by their contribution for node splitting.
        *  OOB permutation importance: Effect of removing variables in predictions and the degree of its correlation with the response variables.

*   **15.3.3 Proximity Plots:**
    *   Proximity Matrix: Accumulating proximity between two observations by averaging over trees.
     *  Multidimensional scaling plots.
    *  Limitations: star shaped plots and difficulties to interpret in some situations.

*   **15.3.4 Random Forests and Overfitting:**
    *   Overfitting: Potential problem in using fully grown trees, with high variance and low bias.
    *   Bias and Variance Tradeoff: Reduction of variance, without compromising the bias.
     * Effect of the parameter m for variable importance and the ability of each tree to model relevant structure.
*   **15.4 Analysis of Random Forests**
    *   **15.4.1 Variance and the De-correlation Effect:**
        *   Theoretical framework to deconstruct the variance of the average estimator based on sampling correlations.
    *   **15.4.2 Bias:**
        *   Bias in random forest and its relation with the bias of the individual trees.
         *  The role of the parameter m and its trade-off between variance and bias.
        *   Comparison with ridge regression in this context.

*   **15.5 Computational Considerations:**
    *   Computational aspects of applying random forest.
     *   Algorithm of building a random forest using the bootstrap technique.
        *   The cost of computations using different settings.

