# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: trans_cat
#     language: python
#     name: trans_cat
# ---

# +
# NOTE: to open this as a jupyter notebook, use jupytext --to .ipynb in your command line.
import numpy as np
import scipy.stats as stats

def bootstrap_binomial_hypothesis_test(dist_1, dist_2, num_simulations=10000):
    '''
    This function takes in two distributions and performs
    multiple bootstrap samples of one datapoint from each distribution.
    This function then marks whether the dist_1 bootstrap value is larger, equal,
    or smaller than the dist_2 bootstrap value. These comparison results are
    tallied across all the bootstrap samples.

    We create two binomial distributions from these bootstrap samples:
    dist_1_binary_results: 1 if sample_1 > sample_2, else 0
    dist_2_binary_results: 1 if sample_2 > sample_1, else 0

    We then compute a z-statistic from these two binomial distributions
    and compute the p-value doing a normal distribution lookup for the
    z_statistic.

    INPUT
        dist_1: a list of numbers
        dist_2: also a list of numbers
        num_simulations: Integer. The number of bootstrap samples to take
            when comparing the two distributions.
    OUTPUT
        p_value: Float. p-value from the hypothesis test that the two binomial
            distributions of bootstrap comparison results have equal means.
    '''
    dist_1_binary_results = []
    dist_2_binary_results = []
    samples_1 = np.random.choice(dist_1, num_simulations)
    samples_2 = np.random.choice(dist_2, num_simulations)

    for i in range(num_simulations):
        if samples_2[i] > samples_1[i]:
            dist_1_binary_results.append(0)
            dist_2_binary_results.append(1)
        else:
            dist_1_binary_results.append(1)
            dist_2_binary_results.append(0)

    # Get the binomial distributions
    n_1, p_1 = num_simulations, sum(dist_1_binary_results) / len(dist_1_binary_results)
    n_2, p_2 = num_simulations, sum(dist_2_binary_results) / len(dist_2_binary_results)

    # Get the test statistic
    # Formula from Dan's answer here: https://stats.stackexchange.com/questions/113602/test-if-two-binomial-distributions-are-statistically-different-from-each-other
    p_hat = ((n_1 * p_1) + (n_2 * p_2)) / (n_1 + n_2)
    z_numerator = p_1 - p_2
    z_denom = np.sqrt(p_hat*(1 - p_hat) * ((1 / n_1) + (1 / n_2)))
    z_statistic = z_numerator / z_denom
    p_value = stats.norm.sf(z_statistic)

    return p_value


# -

d1 = np.random.sample(2000)
d2 = np.random.sample(2500)
p_value = bootstrap_binomial_hypothesis_test(d1, d2, num_simulations=10000)
print(p_value)

d1 = np.random.choice([10,11,12,10,11,12,10,11])
d2 = np.random.choice([9,10,11,12,9,10,11,12])
p_value = bootstrap_binomial_hypothesis_test(d1, d2, num_simulations=10000)
print(p_value)
