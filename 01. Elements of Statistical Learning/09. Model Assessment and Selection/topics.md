7.2 **Bias, Variance and Model Complexity**

*   **Generalization Performance:** A learning method's ability to predict outcomes on independent test data.
*   **Loss Function:** Quantification of the difference between target variable (Y) and predicted value (f(X)), using metrics like squared or absolute error.
*   **Test Error (Generalization Error):** Prediction error on independent test samples, represented by ErrT = E[L(Y, f(X))|T].
*   **Expected Prediction Error (Expected Test Error):** Average prediction error over all randomness, including training set, represented by Err = E[L(Y, f(X))] = E[ErrT].
*   **Training Error:** The average loss over the training sample, used as a simple way to evaluate a model fit.
*   **Qualitative/Categorical Response Modeling:** Probabilities of categorical responses pk(X) and the related misclassification error evaluation.
*   **Log-Likelihood as Loss Function:** Application of log-likelihood as a loss function for various response distributions, e.g., Poisson, gamma, exponential.
*   **Deviance:** A measure related to the log-likelihood, often defined as -2 times the log-likelihood
*   **Model Selection:** The process of choosing the best model among multiple models based on their performance.
*   **Model Assessment:** Estimating the generalization error of a chosen model on new data.
*   **Data Splitting:** Dividing data into training, validation, and test sets for model training, selection, and final assessment.

7.3 **The Bias-Variance Decomposition**

*   **Expected Prediction Error Decomposition:** Breakdown of prediction error into irreducible error, squared bias, and variance, given by Err(x0) = σ² + Bias² (f(xo)) + Var(f(xo)).
*   **Squared Bias:** The difference between the average of our estimate from the true mean, Bias² (f(xo)) =  [Ef(xo) – f(xo)]².
*  **Variance:** The expected squared deviation of the prediction around its mean, Var(f(xo)) = E[f(xo) – Ef(xo)]²
*  **k-Nearest Neighbor Regression Error:** Application of the bias-variance decomposition in k-NN regression, showing how k affects these components,  Err(xo) = σ² +[(f(xo) − Σ f(xi)/k)]² + Σ (f(xi) - Σf(xi)/k)²/k
*   **Linear Model Error:** Decomposition of error in a linear model, showing the influence of parameter vector β, Err(xo) = σ² + [f(xo) – Efp(xo)]² + ||h(xo)||²σ².
*   **Ridge Regression Error:** The test error of ridge regression highlighting how the weights of the variance term differ from linear models: h(x) = X(XTX + αI)x0
*   **Average Squared Bias:** Decomposition of bias in linear models into model and estimation biases,  Exo [f(x0) - Efa(20)]² = Exo [f(x0) - x0Tβ]² + Exo [x0Tβ - E x0Tβa]².
*   **Bias-Variance Tradeoff:** Exploration of the balance between bias and variance, and how it influences model complexity.
*   **Model Space and Closest Fit:** Model space visualization for linear models with the closest fit representing xβ* and variance indicated by a yellow circle.

7.3.1 **Example: Bias-Variance Tradeoff**
*   **Bias-Variance Tradeoff Illustration:** Numerical examples on synthetic data where the behavior of the bias and variance is demonstrated for a regression and a classification problem.
*   **Squared Bias in Classification:** The discussion of how errors on the right side of a classification boundary don't hurt the prediction.
*   **Bias-Variance Interaction:** The discussion of how the interaction between bias and variance affects prediction error differently for squared error loss and misclassification error.

7.4 **Optimism of the Training Error Rate**

*   **Generalization Error (ErrT):**  Prediction error conditional on a specific training set T: ErrT = Ex0,yo[L(Y°, f(X°))|T]
*   **Expected Error (Err):** Average prediction error over all possible training sets: Err = ErEx0,yo [L(Y°, f(Xº))|T]
*   **Training Error (err):**  Average loss over training samples:  err = 1/N ΣL(Yi, f(xi))
*   **Optimism:** The difference between the in-sample error and training error, indicating how well a method fits the data, op = Errin - err.
*  **Average Optimism:** The expected value of the optimism over training sets, given by  ω = Ey (op).
*   **Covariance of Fitted and True Values:** Quantifying how the true error depends on the effect of fitted values on true values, ω = 2/N Σ Cov (Yi, ŷi).
*  **Optimism in Linear Model:** The simplification of the expected error for linear models, Ey (Errin) = Ey (err) + 2dσ²/N
* **Effective Number of Parameters:** How the optimism grows with the number of input parameters and decreases with the training sample size.

7.5 **Estimates of In-Sample Prediction Error**

*   **In-Sample Error Estimates:** The combination of training error and average optimism, Errin  = err + ω.
*   **Cp Statistic:** The squared error loss version of the in-sample estimate, incorporating the number of parameters and variance of the noise, Cp = err + 2dσ²/N.
*   **Akaike Information Criterion (AIC):** Using log-likelihood loss, the estimate of the prediction error, -2 E[log Prθ(Y)] – E[loglik] + 2d/N.
*   **AIC for Gaussian Model:** The AIC statistic reduces to the Cp statistic when assuming gaussian distributions with known variance.
*   **AIC for Model Selection:** The application of the AIC criterion for choosing models with the smallest AIC value.
*  **AIC for nonlinear models:** The need to replace d by a measure of complexity for nonlinear and other models where parameter counts do not translate well into effective parameters.
*   **AIC in Practice:** An illustration of AIC on phoneme recognition, by minimizing Error both for entropy and 0-1 loss.

7.6 **The Effective Number of Parameters**

*   **Linear Fitting Method:** Defining linear methods as those that can be written as ŷ = Sy, where S is the matrix depending on input vectors, not on the outcomes.
*   **Effective Number of Parameters (Degrees-of-Freedom):**  Defining the effective number of parameters, or degrees-of-freedom, as the trace of matrix S, df(S) = trace(S).
*   **Regularization in Linear Models:** Discussing how models that use shrinkage, ridge regression and splines can have effective parameters less than their parameter counts.
* **Effective Parameters for Neural Networks:** Defining the effective parameters in Neural Networks as a weighted sum of the eigenvalues of the Hessian matrix of the weights.

7.7 **The Bayesian Approach and BIC**

*  **Bayesian Information Criterion (BIC):** Application of the BIC criterion, and how it depends on log-likelihood, the number of parameters, and the sample size, BIC = -2loglik + (log N) d.
*  **Schwarz Criterion:** The BIC statistic is also known as the Schwarz criterion.
*  **BIC for Gaussian Model:** The connection between BIC and the squared error loss when using a gaussian model is shown here:  BIC = N/σ² [err + (log N)d/N ].
*  **BIC vs AIC:** Highlighting how BIC penalizes complex models more than AIC.
*  **BIC for Classification:** The use of multinomial log-likelihood in classification problems that leads to a similar result.
*  **Bayesian Model Selection:** Introducing the Bayesian approach for model selection, based on calculating the posterior probability of a model, Pr(Mm|Z) ∝ Pr(Mm)⋅Pr(Z|Mm).
*   **Posterior Odds:** Comparing two models using posterior odds, and Bayes factor, using this formula: Pr(Mm|Z)/Pr(Ml|Z) = Pr(Mm)/Pr(Ml) * Pr(Z|Mm)/Pr(Z|Ml)
*  **Bayes Factor:** The contribution of the data towards the posterior odds: BF(Z) = Pr(Z|Mm)/Pr(Z|Ml).
*   **Laplace Approximation:** Approximating the model probability with a log probability: log Pr(Z|Mm) = log Pr(Z|θ̂m,Mm) - dm/2 log N + O(1).

7.8 **Minimum Description Length**

*   **Minimum Description Length (MDL):** Using optimal coding techniques and defining it as a method for model selection from a data compression point of view.
*   **Data as Message:** Framing the data as a message to be encoded and sent using the most parsimonious model.
*   **Instantaneous Prefix Codes:** A discussion of instantaneous prefix codes where no code is the prefix of any other.
*  **Shannon's Theorem:** The theorem that determines that code lengths should be given by -log Pr(zi), for a probability distribution Pr(zi).
*   **Entropy:** The lower bound of code lengths, where E(length) ≥ − Σ Pr(zi) log2(Pr(zi)).
* **Model Selection by Message Length:** How a model is selected when minimizing message length based on a given dataset,
    length = - log Pr(y|θ, M, X) - log Pr(θ|M).
*   **MDL as Negative Log-Posterior:** Highlighting the equivalence between MDL and maximizing posterior probability.

7.9 **Vapnik-Chervonenkis Dimension**
* **Vapnik-Chervonenkis (VC) Theory:** Introducing VC theory to measure the complexity of a function class and to derive related bounds on optimism.
*   **Indicator Functions:** Indicator functions that return 0 or 1 as the class of functions used for the first examples of VC dimensions.
*   **Shattering:** Defining when a class of functions can shatter a set of points perfectly, regardless of how labels are assigned to those points.
*  **VC Dimension:** Defining the VC dimension as the largest number of points that can be shattered by the function class, even for an arbitrarily worst configuration.
* **VC dimension of linear indicator functions:** Illustrating that the VC dimension of linear indicator functions in the plane is 3 because 3 points can be shattered, but 4 points cannot.
*  **Infinite VC Dimension:** The explanation of why the sin(ax) function has infinite VC dimension.
*  **VC dimension of real valued functions:** Defining VC dimension of real-valued functions, g(x, a), as the VC dimension of the indicator class, {I(g(x, α) – β > 0)}.
* **Prediction Error with VC Dimension:** How VC dimension is used to construct an estimate of extra sample prediction error.
*   **Structural Risk Minimization:** The concept of fitting nested sequences of models, where models are picked according to the smallest upper bound from the VC dimension.
* **Limitations of VC Theory:**  Discussing the difficulty in calculating the VC dimension of a function class and its potential loose bounds.

7.9.1 **Example (Continued)**
*  **VC in model selection:** A study on how AIC, BIC, and SRM are used to select the best model size for the k-NN and Linear regression problem.
* **Comparison of Methods:** Boxplots of the relative error of the models chosen through each model selection procedure.

7.10 **Cross-Validation**

*  **Cross-Validation:** The most widely used method for estimating prediction error based on the average generalization error over independent samples.
*   **K-Fold Cross-Validation:** The splitting of data into K parts to fit the model on K - 1 parts and test it on the other part.
*   **Cross-Validation Estimate of Prediction Error:** Estimating prediction error by using the cross-validation technique, where CV(f) = 1/N Σ L(yi, f-k(i)(xi)).
*   **Leave-One-Out Cross-Validation:** The particular case of K-fold CV where K = N, and for every point a model is fit without that point.
*   **Cross-Validation for Model Selection:** Application of CV for model selection, given a set of models indexed by parameter a, CV(f, a) = 1/N Σ L(yi, f-k(i)(xi, a)).
*   **Bias and Variance Tradeoff in Cross-Validation:** A discussion on how small K values can lead to biased estimates of the prediction error, while K = N can have high variance.
* **Learning curves and Cross-validation**: A discussion on the impact of the slope of the learning curve on the cross-validation estimate.
*   **Generalized Cross-Validation (GCV):** An approximation of Leave-One-Out Cross-Validation for linear models under squared error loss, related to the trace of matrix S, as GCV(f) = NΣ (yi - f(xi))² / (N-trace(S))²
* **Effective Parameters in GCV:** The application of the effective number of parameters on GCV, as defined in section 7.6.
* **Similarity between GCV and AIC:** The observation that GCV and AIC have a similar relationship from the approximation 1/(1 - x)² ≈ 1 + 2x

7.10.2 **The Wrong and Right Way to Do Cross-validation**

*   **Incorrect Cross-Validation Method:** An example of how selecting predictors based on all data, then performing cross-validation on a subset of data, can lead to incorrect error estimates.
*   **Correct Cross-Validation Method:** Emphasizing how to apply cross-validation correctly, using all steps of the procedure at each fold.
*   **Unsupervised Screening:** An exception to the cross-validation principle, where unsupervised steps such as variance calculation can be used before cross-validation is applied.

7.10.3 **Does Cross-Validation Really Work?**
*   **Cross-validation pitfalls:** An analysis on how cross validation can fail in high dimensional data if we don't take into account that the model has to be completely refitted at every fold of cross-validation.
*   **Simulation analysis of CV behavior:** The performance of CV is analyzed in simulation settings, showing that CV is working as intended.
*  **Variability of CV error estimates:** Highlighting the importance of reporting CV's standard errors, since variability can be very high.

7.11 **Bootstrap Methods**

*  **Bootstrap resampling:** Introduction of the bootstrap method, resampling with replacement from the training dataset to assess statistical accuracy.
*   **Bootstrap Datasets:** How a bootstrap method uses B bootstrap samples from the original dataset to replicate and examine the fit.
*   **Bootstrap Variance Estimation:**  Estimating the variance of a quantity computed from the bootstrap samples with Var[S(Z)] = 1/(B-1) Σ(S(Z*b)-S*)².
* **Bootstrap for Prediction Error**: Estimating prediction error by using the fitted model on the bootstrap sample as the training data, and the original dataset as test data.
* **Limitations of Bootstrap Prediction Error:** How the bootstrap estimator can overfit the prediction error if it's not adjusted to mimic cross-validation.
* **Leave-One-Out Bootstrap:** Solving the overfitting of standard bootstrap by using only the predictions from bootstrap samples that do not contain the tested observation: Err = 1/N Σ 1/|C-i| Σ L(yi, fb*(xi))
*   **".632 Estimator":** Deriving and explaining the ".632 estimator", which pulls the bootstrap estimator down toward the training rate to correct for its upward bias, err(.632) = 0.368err + 0.632err(1)
*  **Relative Overfitting Rate:** Introduction of the relative overfitting rate, which ranges from 0 to 1 and serves as a measure to compute the adjusted bootstrap method: R = (err(1) - err)/ (ŷ - err)
*   **".632+ Estimator":** Final definition of the ".632+" bootstrap estimator,  Erra^(0.632+) = (1-ω)err + ωerr(1).

7.11.1 **Example (Continued)**

* **Comparison of methods:** The .632+ bootstrap method and 10-fold cross validation, being used as model selection techniques in a specific simulation study.
*   **Model Selection and Bias:** Emphasizing that for model selection, the methods can be biased if the relative performance of the models are not affected.
*  **Limitations of AIC:** Discussing why methods such as AIC may be impractical for adaptive non-linear techniques.

7.12 **Conditional or Expected Test Error?**

*   **Conditional vs Expected Error:** Addressing the difference between conditional test error, Errt, and expected test error, Err.
*  **Cross-Validation for Conditional Error:** How both leave-one-out and 10-fold cross validation are expected to estimate the conditional test error.
*   **Simulation Study:** A simulation to study the behavior of cross-validation in estimating both the conditional and expected errors.
* **10-Fold and N-Fold Estimators:** An analysis on how the 10-fold cross-validation has lower variance than the leave-one-out in estimating conditional test error.
*   **Negative Correlation:** The surprising observation that CV estimates of the error are negatively correlated with the conditional error.
* **Importance of a Separate Test Set**: Highlighting why a separate test set may be needed in some situations, due to the high variability of the conditional test error and bias associated with cross-validation.
