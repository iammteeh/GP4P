import numpy as np
from scipy.stats import normaltest, kstest, anderson, shapiro

# generate some random data
data = np.random.normal(0, 1, 1000)

# test for normal distribution using the D'Agostino-Pearson test
stat, p = normaltest(data)
print("D'Agostino-Pearson normality test p-value:", p)

# test for uniform distribution using the Kolmogorov-Smirnov test
stat, p = kstest(data, 'uniform')
print("Kolmogorov-Smirnov uniformity test p-value:", p)

# test for exponential distribution using the Anderson-Darling test
result = anderson(data, 'expon')
print("Anderson-Darling exponentiality test statistic:", result.statistic)
print("Anderson-Darling exponentiality test critical values:", result.critical_values)
print("Anderson-Darling exponentiality test significance levels:", result.significance_level)

# test for normal distribution using the Shapiro-Wilk test
stat, p = shapiro(data)
print("Shapiro-Wilk normality test p-value:", p)