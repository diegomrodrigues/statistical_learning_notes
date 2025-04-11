*   **16.1 Introduction:**
    *   Core Idea: Combining multiple base learners to create more accurate and robust prediction models.
        * Emphasizes on a broad range of methods: Bagging, Random Forests, Boosting, Stacking, Basis expansions and Bayesian approaches.
    *   Ensemble Approach:  Two phases for ensemble learning: generation of base learners and combination of these learners.
     * Boosting as Supervised Search:  Building an ensemble with a guided search of weak learners in a high-dimensional model space.
       *   Relation to dictionary based methods.

*   **16.2 Boosting and Regularization Paths**
    *   Analogy: Parallels drawn between the sequence of models from gradient boosting and regularized linear model fitting.
    *   **16.2.1 Penalized Regression:**
        *   Gradient boosting and L₁ regularization through an approximation in high dimensional feature spaces via a dictionary of basis functions.
        *  Relationship between boosting and lasso, and the interpretation of boosting with shrinkage as a monotone path.
    *   **16.2.2 The “Bet on Sparsity” Principle**
        *  The sparsity principle in high dimensional spaces and the benefits of L₁ regularization over L₂ regularization in these scenarios.
   
*   **16.3 Learning Ensembles**
    *   General Framework: Breaking down ensemble learning into developing a population of base learners, and then combining those in order to create a composite predictor.

 * **16.3.1 Learning a Good Ensemble**
    *   Post-Processing: improving the ensemble via lasso-post processing, that results in improved performance.
    *    The need for a method that spans the space well, is able to capture local structures, and that is well suited for a high-dimensional space of learners.
*   **16.3.2 Rule Ensembles:**
   *   Rule derivation from trees: Deriving a set of rules from every tree to enlarge the ensemble, and make an automatic combination of features
    * Use of a meta model to combine the rules and find a good path for the solution.
*   **16.4 Stochastic Gradient Boosting**
   *   An alternative approach via randomization of models, and a more efficient selection of a set of parameters.
