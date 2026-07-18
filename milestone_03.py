```python
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import (
    proportion_confint,
    proportions_ztest
)

# ===================================================
# LOAD DATA
# ===================================================

try:
    df = pd.read_csv("data/finflow_users.csv")
    ab_df = pd.read_csv("data/finflow_ab_test.csv")
except FileNotFoundError as e:
    print(f"Dataset missing: {e}")
    raise

# ===================================================
# PART 1: CONFIDENCE INTERVALS
# ===================================================

n = len(df)

mean_minutes = df["session_minutes"].mean()
sd_minutes = df["session_minutes"].std(ddof=1)

se_minutes = sd_minutes / np.sqrt(n)

t_crit = stats.t.ppf(0.975, df=n - 1)

margin_error_mean = t_crit * se_minutes

ci_mean_lower = mean_minutes - margin_error_mean
ci_mean_upper = mean_minutes + margin_error_mean

# Wilson Interval

successes = df["premium_user"].sum()

p_hat = successes / n

ci_prop_lower, ci_prop_upper = proportion_confint(
    successes,
    n,
    alpha=0.05,
    method="wilson"
)

margin_error_prop = (
    ci_prop_upper - ci_prop_lower
) / 2

# ===================================================
# PART 2: BOOTSTRAP CI
# ===================================================

session_minutes = df["session_minutes"].values

n_boot = 10000

bootstrap_medians = np.zeros(n_boot)

for i in range(n_boot):

    sample = np.random.choice(
        session_minutes,
        size=len(session_minutes),
        replace=True
    )

    bootstrap_medians[i] = np.median(sample)

point_estimate_median = np.median(
    session_minutes
)

ci_boot_lower = np.percentile(
    bootstrap_medians,
    2.5
)

ci_boot_upper = np.percentile(
    bootstrap_medians,
    97.5
)

# ===================================================
# PART 3: T TEST
# ===================================================

free_users = df[
    df["premium_user"] == 0
]["session_minutes"]

premium_users = df[
    df["premium_user"] == 1
]["session_minutes"]

h0_ttest = (
    "H0: μpremium ≤ μfree"
)

ha_ttest = (
    "Ha: μpremium > μfree"
)

_, shapiro_free_p = stats.shapiro(
    free_users
)

_, shapiro_premium_p = stats.shapiro(
    premium_users
)

normality_ok = (
    shapiro_free_p > 0.05
    and
    shapiro_premium_p > 0.05
)

_, levene_p = stats.levene(
    free_users,
    premium_users
)

equal_var = levene_p > 0.05

t_stat, p_two_tail = stats.ttest_ind(
    premium_users,
    free_users,
    equal_var=False
)

if t_stat > 0:
    p_value_ttest = p_two_tail / 2
else:
    p_value_ttest = 1 - p_two_tail / 2

reject_h0_ttest = (
    p_value_ttest < 0.05
)

# Cohen d

n1 = len(free_users)
n2 = len(premium_users)

sd1 = free_users.std(ddof=1)
sd2 = premium_users.std(ddof=1)

pooled_sd = np.sqrt(
    (
        ((n1 - 1) * sd1 ** 2)
        +
        ((n2 - 1) * sd2 ** 2)
    )
    /
    (n1 + n2 - 2)
)

cohens_d = (
    premium_users.mean()
    -
    free_users.mean()
) / pooled_sd

n_needed_ttest = np.ceil(
    16 / (cohens_d ** 2)
)

# ===================================================
# PART 4: CHI SQUARE
# ===================================================

contingency_table = pd.crosstab(
    df["risk_profile"],
    df["premium_user"]
)

chi2_stat, p_value_chi2, dof_chi2, expected = (
    stats.chi2_contingency(
        contingency_table
    )
)

min_expected = expected.min()

assumption_met = (
    min_expected >= 5
    or
    (
        np.sum(expected >= 5)
        / expected.size >= 0.80
        and
        min_expected >= 1
    )
)

n_chi2 = contingency_table.sum().sum()

cramers_v = np.sqrt(
    chi2_stat
    /
    (
        n_chi2
        *
        min(
            contingency_table.shape[0] - 1,
            contingency_table.shape[1] - 1
        )
    )
)

# ===================================================
# PART 5: A/B TEST
# ===================================================

conversion_rates = (
    ab_df.groupby("variant")
    ["converted"]
    .mean()
)

control_rate = conversion_rates[
    "control"
]

variants_to_test = [
    "variant_a",
    "variant_b",
    "variant_c",
    "variant_d"
]

alpha = 0.05
alpha_adj = alpha / 4

results = []

for variant in variants_to_test:

    variant_data = ab_df[
        ab_df["variant"] == variant
    ]

    control_data = ab_df[
        ab_df["variant"] == "control"
    ]

    count = np.array([
        variant_data["converted"].sum(),
        control_data["converted"].sum()
    ])

    nobs = np.array([
        len(variant_data),
        len(control_data)
    ])

    _, p_value_ab = proportions_ztest(
        count,
        nobs,
        alternative="larger"
    )

    variant_rate = conversion_rates[
        variant
    ]

    abs_lift = (
        variant_rate
        -
        control_rate
    )

    rel_lift = (
        abs_lift
        /
        control_rate
    )

    results.append({

        "variant": variant,
        "conversion_rate":
            variant_rate,

        "p_value":
            p_value_ab,

        "significant":
            p_value_ab
            < alpha_adj,

        "abs_lift":
            abs_lift,

        "rel_lift":
            rel_lift
    })

results_df = pd.DataFrame(
    results
)

# ===================================================
# VALIDATION
# ===================================================

assert (
    ci_mean_lower
    <
    mean_minutes
    <
    ci_mean_upper
)

assert (
    ci_prop_lower
    <
    p_hat
    <
    ci_prop_upper
)

assert (
    ci_boot_lower
    <
    point_estimate_median
    <
    ci_boot_upper
)

assert cohens_d > 0
assert 0 <= p_value_chi2 <= 1
assert len(results_df) == 4

# ===================================================
# OUTPUT
# ===================================================

print("\nCONFIDENCE INTERVALS")
print("=" * 70)

print(
    f"Mean Session Duration:"
    f" ({ci_mean_lower:.2f},"
    f" {ci_mean_upper:.2f})"
)

print(
    f"Premium Conversion:"
    f" ({ci_prop_lower:.2%},"
    f" {ci_prop_upper:.2%})"
)

print(
    f"Median Session:"
    f" ({ci_boot_lower:.2f},"
    f" {ci_boot_upper:.2f})"
)

print("\nT TEST")
print("=" * 70)

print(
    f"t={t_stat:.3f}"
)

print(
    f"p={p_value_ttest:.4f}"
)

print(
    f"Cohen d={cohens_d:.3f}"
)

print(
    f"Required n="
    f"{n_needed_ttest:.0f}"
)

print("\nCHI SQUARE")
print("=" * 70)

print(
    f"Chi2={chi2_stat:.3f}"
)

print(
    f"p={p_value_chi2:.4f}"
)

print(
    f"Cramers V="
    f"{cramers_v:.3f}"
)

print("\nA/B RESULTS")
print("=" * 70)

print(results_df)
```


