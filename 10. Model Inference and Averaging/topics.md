8.2 **The Bootstrap and Maximum Likelihood Methods**

8.2.1 **A Smoothing Example**
*   **Bootstrap for Uncertainty Assessment:** Using resampling to quantify uncertainty in model predictions through a simple smoothing example.
*  **B-Spline Basis Functions:** The use of linear expansion of B-splines to represent function spaces.
*  **Least Squares Estimates of Parameters:**  The minimization of squared errors in order to obtain the parameter estimates:  β = (HTH)-1HTy
*  **Covariance Matrix:** The calculation of the covariance matrix of beta using (HTH)-1σ²
* **Standard error:** The use of standard error to estimate confidence bands on the function being fitted using se[μ(x)] = [h(x)T(HTH)-1h(x)]σ.
*  **Nonparametric Bootstrap:** Sampling with replacement from training data to produce new datasets for uncertainty estimation.
*   **Parametric Bootstrap:** Simulating new responses by adding Gaussian noise to predicted values, instead of resampling from the training data, y*i = μ(xi) + εi

8.2.2 **Maximum Likelihood Inference**

*   **Parametric Model:** Defining a parametric model as a probability density or mass function for our data: zi ~ gθ(z)
*   **Likelihood Function:**  The likelihood of the model given parameters, defined as:  L(θ;Z) = ∏ gθ(zi)
*   **Log-Likelihood:** The log of the likelihood, defined as:  l(θ;Z) = ∑ log gθ(zi)
* **Maximum Likelihood:**  Defining the maximum likelihood as the method that chooses the parameters that maximize the log-likelihood function: 𝜃 =argmax l(θ;Z)
*   **Score Function:** The derivative of the log-likelihood with respect to parameters, ℓ(0; Z) = Σ ℓ(θ; zi).
*   **Observed Information:** The negative second derivative of the log-likelihood evaluated at the maximum likelihood estimates: I(θ) = - ∑ d²l(θ; zi)/dθdθT.
*   **Fisher Information:** The expected value of the observed information: i(θ) = E[I(θ)].
*   **Maximum Likelihood Estimator Distribution:** The limiting distribution of the maximum likelihood estimator that follows a normal distribution as N tends to infinity: θ ~ N(θ0, i(θ0)-1)
*   **Standard Error of Maximum Likelihood Estimator:** The square root of the inverse of the observed information matrix: √I(θ)-1
*   **Confidence Intervals:** Constructing confidence intervals for the parameters by using the z score of the normal distribution, or by using a chi-squared approximation,  2[l(θ) – l(θ0)] ~ χ²p

8.2.3 **Bootstrap versus Maximum Likelihood**

*   **Maximum Likelihood in Smoothing:** Showing the process for how the maximum likelihood works for the B-spline smoothing, as well as how the maximum likelihood parameters matches the linear estimates.
*  **Bootstrap as a computational implementation**: The use of the bootstrap as a computer implementation of parametric or non parametric maximum likelihood, enabling computation of standard error estimates.
*   **Adaptive Choices of Parameters**: Discussing how bootstrap enables the consideration of uncertainty in adaptive choices of parameters, such as the number of knots.
*   **Bootstrap confidence bands:** The explanation of how bootstrap methods can be used in situations where formulas for standard errors and confidence intervals are difficult or not available.

8.3 **Bayesian Methods**

* **Bayesian Approach:** The specification of a sampling model Pr(Z|θ), and a prior distribution, Pr(θ), in order to calculate a posterior distribution Pr(θ|Z).
*  **Posterior Distribution:** Calculating the posterior distribution as the updated knowledge about the parameters after seeing the data: Pr(θ|Z) = Pr(Z|θ).Pr(θ) / ∫ Pr(Z|θ).Pr(θ)dθ.
*   **Predictive Distribution:** The posterior distribution basis for predicting the values of a future observation znew.
    Pr(znew|Z) = ∫ Pr(znew|θ).Pr(θ|Z)dθ.
*   **Prior Distributions:** Emphasizing that Bayesian methods use prior distributions that allow for the expression of uncertainty before data is observed.
*   **Gaussian Process Prior:** The use of Gaussian processes to specify a prior in the function space, specifying the covariance between values:   K(x,x’) = τ cov [μ(x), μ(x’)]
*   **Gaussian Prior:** A simpler alternative to the gaussian process, that specifies a gaussian prior on the function coefficients, β ~ N(0, τ∑).
* **Posterior Distribution Computation:** The computation of the posterior distribution for the coefficients when using a gaussian prior, by E(β|Z) = (HTH + (σ²/τ)∑)-1 H¹y and cov(β|Z) = σ² (HTH + (σ²/τ)∑)-1.
* **Bayesian Posterior Function values**: The posterior distribution for the values of the function, based on the coefficients posterior values: E(μ(x)|Z) = h(x)T (HTH + (σ²/τ)∑)-1 H¹y, and cov[μ(x), μ(x')|Z] = h(x)T (HTH + (σ²/τ)∑)-1 h(x') σ².
*  **Prior Correlation Matrix (Σ):** The specification of the prior correlation matrix Σ and its impact on the posterior results.
*   **Noninformative Prior:** The use of a noninformative prior, where τ → ∞, making the posterior proportional to the likelihood.
*   **Prior Variance:** The discussion of how the prior variance, τ, controls the smoothness of the model.
*  **Joint posterior:** Highlighting that, while in this text it is not done, one should put a prior on σ, and compute the joint posterior distribution.

8.4 **Relationship Between the Bootstrap and Bayesian Inference**

*   **Simple Example for Bayesian Inference:** A simple example where we observe one point from a normal distribution z ~ N(θ,1), where the prior distribution for θ is θ ~ N(0,τ).
*  **Posterior with Gaussian Prior:**  The posterior distribution of theta in the simple example, θ|z ~ N(z/(1+1/τ), 1/(1+1/τ)).
*  **Noninformative Prior:** The limit of the posterior distribution with a noninformative prior, where the variance τ goes to infinity, that is, θ|z ~ N(z, 1).
*  **Parametric Bootstrap Connection:** The equivalence of the posterior distribution with a noninformative prior to parametric bootstrap with maximum likelihood.
* **Three Key Ingredients for Correspondence:** Highlighting that three ingredients make this correspondence work: noninformative prior, dependence on maximum likelihood estimate, and symmetry of the log likelihood.
*  **Nonparametric Bootstrap and Bayes Correspondence:** The outline of the correspondence of nonparametric bootstrap and Bayes inference in a multinomial distribution.
* **Dirichlet Distribution:** The use of a Dirichlet distribution as a prior on a multinomial sample space with parameters w: w ~ Dir(αI)
* **Posterior Dirichlet Distribution**: The posterior density for the prior based on samples,  w ~ Dir(αI + Nŵ), and how it tends to a bootstrap-like distribution when α goes to 0, w ~ Dir(Nŵ).
*   **Bootstrap Distribution as Posterior:** Framing the bootstrap as an approximate noninformative posterior, that is obtained by perturbing data.

8.5 **The EM Algorithm**

*   **EM Algorithm Introduction:** The discussion of the EM algorithm as a tool to simplify maximum likelihood problems, specifically on mixture models.
*   **Mixture Model:** An introduction to a simple mixture model, where the response variable Y can come from one of two normal distributions with different parameters,  Y ~ (1 − Δ)Y1 + ΔY2.
*   **Log Likelihood Mixture Model:** The log-likelihood of a mixture model for N training cases: l(θ;Z) = ∑ log[(1 − π)φθ1(yi) + πφθ2(yi)].
*   **Latent Variables:** Highlighting how the EM algorithm works by introducing a latent unobserved variable Δ that indicates where the data comes from.
*  **Complete Data Log-Likelihood:** The use of the complete data log-likelihood using the latent variable, l(θ;Z,Δ) = ∑ [(1 – Δi) log φθ1(yi) + Δi log φθ2(yi)] + ∑ [(1 – Δi) log(1 – π) + Δi log π].
*   **Responsibility:** How the EM algorithm computes the expected values of the latent variables, called responsibility,  γi(θ) = E(Δi|θ, Z) = Pr(Δi = 1|θ, Z).
*   **Expectation Step:** The step where we compute the responsibilities using the current estimates of the parameters.
*   **Maximization Step:**  The step where we re-estimate parameters using the responsibilities.
*  **Initial Guesses:** The explanation of how to construct initial parameters for the algorithm.
*   **EM convergence:** The explanation that the EM algorithm only converges to a local maximum, with several local maxima possible.

8.5.2 **The EM Algorithm in General**

*  **Data Augmentation:** Emphasizing that the EM algorithm is useful in problems where maximization is made easier by enlarging the dataset.
*   **Complete Data:** The distinction between observed data Z and the latent or missing data Zm, making T=(Z, Zm) the complete data.
*  **Expectation Step (General):** The computation of the expected value of the complete data log likelihood,  Q(θ', θ(i)) = E(l0(θ'; T)|Z, θ(i)).
*  **Maximization Step (General):** The re-estimation of the parameters by maximizing Q(θ', θ(i)) with respect to θ': 𝜃(i+1) = argmax Q(θ', θ(i))
*   **EM Convergence:**  Proof that the EM algorithm never decreases the log-likelihood using Jensen's inequality.
*   **Generalized EM (GEM):** The explanation that the maximization step can only increase Q(θ', θ(i)), not necessarily find a global maximum.

8.5.3 **EM as a Maximization-Maximization Procedure**

* **Joint Maximization:** Describing a different approach to the EM procedure as a joint maximization for the objective function F(θ', P) = Ep[lo(θ'; T)] – Ep[log P(Zm)].
* **Maximization over Latent Space:** The description of the E step as the maximization of F with respect to the latent variables P.
* **Maximization over Parameter Space:** The description of the M step as the maximization of F with respect to the model parameters, θ.
*  **Alternative Maximization Procedure**: How viewing the EM algorithm as a joint maximization procedure leads to alternative optimization procedures.

8.6 **MCMC for Sampling from the Posterior**
*  **MCMC Introduction:** The introduction of Markov Chain Monte Carlo (MCMC) approaches as a powerful way for sampling from a posterior distribution.
*   **Gibbs Sampling:** The explanation of how a Gibbs sampler generates a sequence of correlated samples by sampling each variable from its conditional distributions,   Pr(Uk|U1, U2, ..., Uk-1, Uk+1, ...,UK)
*  **Gibbs Sampling Convergence**: How a Gibbs sampler produces samples from the joint distribution when the process stabilizes, despite the lack of independence.
*   **Estimation with Gibbs Samples:**  Estimating marginal distributions with a density estimate applied to the Gibbs sample.
*   **Gibbs Sampling for Bayesian Inference:** Application of Gibbs sampling to generate samples from the joint posterior distribution of model parameters, conditional on the data: Pr(θ|Z)
*   **Gibbs Sampling in Gaussian Mixtures:** Application of Gibbs sampling to the gaussian mixture problem by fixing σ1, σ2 and π and sampling μ1, μ2 and  Δi.
*   **Gibbs Sampling Steps:** The explanation that each Gibbs step, is equivalent to the E and M steps in an EM algorithm, but the parameters are being sampled instead of maximized.
* **Gibbs Sampling Behavior:** A simulation result showing how the parameters seem to stabilize at the correct values.
*  **Proper Priors**:  The need for proper (informative) priors in Gibbs sampling, in order to avoid a degenerate posterior.
*   **Gibbs Sampling is Conditional Sampling:**  Highlighting that Gibbs sampling uses conditional sampling given the rest, making this method suitable for problems where this is easy to carry out.

8.7 **Bagging**
*   **Bootstrap Aggregation (Bagging):** Using the bootstrap to improve prediction by averaging it over a collection of bootstrap samples, thereby reducing the variance of the predictions:  fbag(x) = 1/B  ∑ f*b(x)
*   **Bagging Estimates:**  Defining the bagged estimate in terms of a real population distribution P as fbag(x) = Epf*(x)
*   **Bagging for Regression:** Describing how Bagging is obtained in regression, and how bagged prediction may only differ from original prediction in adaptive and nonlinear functions.
*   **Bagging for Classification:**  Defining the use of bagging in classification with an underlying indicator vector, defining the bagged classifier as a combination of single classifiers.
*   **Averaging Class Probabilities:**  Highlighting how it might be better to average class probabilities rather than using a “vote” rule.
*  **Bagging and Weak Learners:** How the concept of “Wisdom of Crowds” can be applied to bagging, where a consensus of independent weak learners can produce better prediction.
*   **Loss of Model Structure:** Highlighting how bagging a model might mean losing interpretability in certain cases.
*   **Bagging Limitations:** Emphasizing that bagging might not be the best approach for all problems, especially when using a very simple base learner.
*   **Trees with Simulated Data**: A simulation study of bagging on trees, showing the effects of bagging in variance reduction for tree classifiers.

8.7.1 **Example: Trees with Simulated Data**

*   **Simulation Scenario:** Generating synthetic data with two classes and five correlated features in order to showcase bagging on decision trees.
*  **Variance Reduction in Bagging:** Showing that bagging can reduce the test error of unstable classifiers like decision trees.
*   **Original vs. Bagged Trees:** Visual illustration of differences between the original and bagged trees.
*   **Test Error Curves:** Curves for bagged decision trees, and consensus vote, showing reduction in variance as the number of trees grows.

8.8 **Model Averaging and Stacking**

*   **Bayesian Model Averaging:**  A method where bootstrap values are used as a posterior sample, with the bagged estimate as an approximation of the posterior mean.
*  **Model Averaging:** The process of combining predictions from different candidate models by weighing them by the posterior probabilities of each model: E(ζ|Z) = Σ E(ζ|Mm, Z)Pr(Mm|Z)
*   **Committee Methods:** Taking a simple unweighted average of model predictions.
*  **BIC for Model Averaging**:  How the BIC criterion can be used to estimate the posterior probability of different models.
*  **Bayesian Approach to Model Averaging:** A detailed description on how to do model averaging, using a proper prior and numerically calculating posterior probabilities: Pr(Mm|Z) = Pr(Mm) ∫ Pr(Z|θm, Mm)Pr(θm|Mm)dθm
*   **Frequentist Viewpoint of Model Averaging:** A description on how to calculate weights for model averaging by minimizing the squared-error loss, argmin Ep[Y - ∑wmfm(x)]².
* **Linear Regression in Model Averaging**:  How the weights can be found by a linear regression of the outcome with respect to the predictions of the different models:  w = Ep[F(x)F(x)T]¯¹Ep[F(x)Y].
*   **Stacking:**   A technique where the training set linear regression is replaced by leave-one-out cross-validation estimates of the predictions.
*  **Stacking for Model Selection:**  How stacking can be used as an approximation to choosing only the best model with cross-validation, but providing a better performance.
*  **Flexibility of Stacking**: Highlighting how stacking is more flexible, allowing a combination of different learning methods.

8.9 **Stochastic Search: Bumping**
*   **Bumping Introduction:**  The explanation of bumping as an alternative approach to finding a good model by avoiding poor local minima.
*  **Bootstrap Samples for Bumping**: How bumping uses bootstrap samples in order to search the model space.
*   **Bumping Algorithm:** Detailed description of how the bootstrap method is applied to find a better model, by choosing a bootstrap estimate that minimizes the prediction error:  b* = argmin  ∑  [yi - f*b(xi)]²
*  **Bumping for Tree Models**: A discussion on why the method is useful for tree-based models, as well as other methods that have unstable procedures due to high variability.
*   **Illustration of Bumping on Tree Models:** The use of a XOR example where standard tree growing procedure does not have an optimal split, and how bumping can help find a better split.
*  **Bumping and Model Complexity**: That when doing bumping, the models must have roughly the same complexity.
*  **Bumping and Optimization**:  The use of bumping when the fitting method is hard to optimize, or lacks smoothness, by optimizing a different criterion.
