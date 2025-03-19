import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact

def test_continuous_feature(data, feature, target='target', alpha=0.05):
    """
    Performs hypothesis testing on a continuous feature.
    
    First, it checks for normality using the Shapiro-Wilk test.
    If both groups are normally distributed, it uses an independent t-test;
    otherwise, it uses the Mann-Whitney U test.
    
    Parameters:
        data (pd.DataFrame): The dataset.
        feature (str): The name of the continuous feature.
        target (str): The name of the binary target variable.
        alpha (float): Significance level for normality test.
        
    Returns:
        dict: A dictionary with the test used, statistic, and p-value.
    """
    # Separate the feature values by target class
    group0 = data[data[target] == 0][feature].dropna()
    group1 = data[data[target] == 1][feature].dropna()
    
    # Check for normality
    p_normal_group0 = shapiro(group0)[1]
    p_normal_group1 = shapiro(group1)[1]
    
    if p_normal_group0 > alpha and p_normal_group1 > alpha:
        # Both groups look normal, use t-test
        stat, p_val = ttest_ind(group0, group1, equal_var=False)
        test_used = 'Independent t-test'
    else:
        # Use non-parametric Mann-Whitney U test if normality is not met
        stat, p_val = mannwhitneyu(group0, group1, alternative='two-sided')
        test_used = 'Mann-Whitney U test'
    
    return {
        'feature': feature,
        'test_used': test_used,
        'statistic': stat,
        'p_value': p_val
    }

def test_categorical_feature(data, feature, target='target'):
    """
    Performs hypothesis testing on a categorical feature.
    
    For a 2x2 table with low expected counts, it uses Fisher's exact test.
    Otherwise, it uses the Chi-square test.
    
    Parameters:
        data (pd.DataFrame): The dataset.
        feature (str): The name of the categorical feature.
        target (str): The name of the binary target variable.
        
    Returns:
        dict: A dictionary with the test used and p-value.
    """
    # Build contingency table
    contingency = pd.crosstab(data[feature], data[target])
    
    # Decide test based on table shape and expected frequencies
    if contingency.shape == (2, 2):
        # Check minimum cell count
        if contingency.values.min() < 5:
            stat, p_val = fisher_exact(contingency)
            test_used = "Fisher's Exact test"
        else:
            chi2, p_val, _, _ = chi2_contingency(contingency)
            stat = chi2
            test_used = 'Chi-square test'
    else:
        chi2, p_val, _, _ = chi2_contingency(contingency)
        stat = chi2
        test_used = 'Chi-square test'
    
    return {
        'feature': feature,
        'test_used': test_used,
        'statistic': stat,
        'p_value': p_val,
        'contingency_table': contingency
    }

def hypothesis_test_feature(data, feature, target='target', feature_type=None):
    """
    Wrapper function to run the appropriate hypothesis test on a given feature.
    
    Parameters:
        data (pd.DataFrame): The dataset.
        feature (str): The feature to test.
        target (str): The binary target variable.
        feature_type (str or None): 'continuous' or 'categorical'.
            If None, the function will infer based on the data type.
    
    Returns:
        dict: The results of the hypothesis test.
    """
    # Infer feature type if not provided
    if feature_type is None:
        if pd.api.types.is_numeric_dtype(data[feature]):
            feature_type = 'continuous'
        else:
            feature_type = 'categorical'
    
    if feature_type == 'continuous':
        return test_continuous_feature(data, feature, target)
    elif feature_type == 'categorical':
        return test_categorical_feature(data, feature, target)
    else:
        raise ValueError("feature_type must be either 'continuous' or 'categorical'.")

# Example usage:
if __name__ == '__main__':
    # Example: Create a synthetic dataset
    import numpy as np
    np.random.seed(42)
    
    # Create a DataFrame with a continuous feature and a categorical feature
    n_samples = 500
    df = pd.DataFrame({
        'feature_cont': np.concatenate([
            np.random.normal(0, 1, int(n_samples*0.95)),  # non-fraud
            np.random.normal(0.5, 1, int(n_samples*0.05))   # fraud
        ]),
        'feature_cat': np.concatenate([
            np.random.choice(['A', 'B'], int(n_samples*0.95)),
            np.random.choice(['A', 'B'], int(n_samples*0.05))
        ]),
        'target': np.concatenate([
            np.zeros(int(n_samples*0.95), dtype=int),
            np.ones(int(n_samples*0.05), dtype=int)
        ])
    })
    
    # Test a continuous feature
    cont_result = hypothesis_test_feature(df, 'feature_cont')
    print("Continuous Feature Test:")
    print(cont_result)
    
    # Test a categorical feature
    cat_result = hypothesis_test_feature(df, 'feature_cat', feature_type='categorical')
    print("\nCategorical Feature Test:")
    print(cat_result)
