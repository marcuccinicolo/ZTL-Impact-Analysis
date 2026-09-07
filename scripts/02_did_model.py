# DiD

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

POLICY_YEAR = 2015

df = pd.read_csv("data/dataset_pm10.csv")

# Post is a pure time indicator: it must be 1 for EVERY city from the policy
# year onwards, treated or not. Deriving it here rather than trusting the column
# keeps Post and DiD from collapsing into the same variable — if Post were 1
# only for treated cities after 2015 it would already BE the interaction, the
# two regressors would be perfectly collinear, and OLS would silently split one
# effect across two identical coefficients.
df['Post'] = (df['Anno'] >= POLICY_YEAR).astype(int)

# BASIC DID
df['DiD'] = df['Treated'] * df['Post']

# DiD regression
model = smf.ols('PM10 ~ Treated + Post + DiD', data=df).fit()
print("=== Basic DiD Regression ===")
print(model.summary())

# TWO-WAY FIXED EFFECTS
# City fixed effects absorb time-invariant differences between cities, year
# fixed effects absorb shocks common to all of them (weather, engine standards,
# the 2020 lockdown). Errors are clustered by city because a city's PM10 is
# correlated with itself over time. This is the headline estimate.
fe_model = smf.ols(
    'PM10 ~ DiD + C(Città) + C(Anno)', data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['Città']})

ci_low, ci_high = fe_model.conf_int().loc['DiD']
print("\n=== DiD with city and year fixed effects (clustered by city) ===")
print(f"Effect of ZTL on PM10: {fe_model.params['DiD']:.3f} ug/m3 "
      f"(se {fe_model.bse['DiD']:.3f}, p {fe_model.pvalues['DiD']:.4f}, "
      f"95% CI [{ci_low:.2f}, {ci_high:.2f}])")
print(f"Clusters: {df['Città'].nunique()} cities — few enough that these "
      f"standard errors deserve caution.")

# DYNAMIC DID 
import numpy as np
from matplotlib.lines import Line2D

df['Anno'] = df['Anno'].astype(int)
df['Città'] = df['Città'].astype(str)
df['Treated'] = df['Treated'].astype(int)
df = df.dropna(subset=['PM10','Treated','Anno'])


# CREATE RELATIVE YEAR (event time)
policy_year = 2015
df['t_rel'] = df['Anno'] - policy_year
df['t_rel'] = df['t_rel'].astype('category')
# baseline = year before treatment
baseline = -1
df['t_rel'] = df['t_rel'].cat.reorder_categories(sorted(df['t_rel'].cat.categories), ordered=True)


# EVENT-STUDY REGRESSION
model = smf.ols(
    "PM10 ~ Treated:t_rel + C(Città) + C(Anno)",
    data=df
).fit(cov_type="cluster", cov_kwds={'groups': df['Città']})

# Extract coefficients
event_terms = [c for c in model.params.index if "Treated:" in c]
event_df = pd.DataFrame({
    "t_rel": [int(term.split("t_rel")[1].strip("[]")) for term in event_terms],
    "coef": [model.params[term] for term in event_terms],
    "se": [model.bse[term] for term in event_terms]
})
event_df = event_df.sort_values("t_rel")
event_df["ci_low"] = event_df["coef"] - 1.96 * event_df["se"]
event_df["ci_high"] = event_df["coef"] + 1.96 * event_df["se"]


# CENTER PRE-TREATMENT COEFFICIENTS
pre_treat_mask = event_df['t_rel'] < 0
pre_mean = event_df.loc[pre_treat_mask, 'coef'].mean()
event_df['coef_adj'] = event_df['coef'] - pre_mean
event_df['ci_low_adj'] = event_df['ci_low'] - pre_mean
event_df['ci_high_adj'] = event_df['ci_high'] - pre_mean


# PLOT
plt.figure(figsize=(10,6))

colors = ["blue" if t < 0 else "orange" for t in event_df["t_rel"]]

# error bars colored
for i, row in event_df.iterrows():
    plt.errorbar(row["t_rel"], row["coef_adj"],
                 yerr=1.96*row["se"],
                 fmt='o',
                 color=colors[i],       
                 ecolor=colors[i],      
                 elinewidth=2,
                 capsize=4)


plt.axhline(0, color='black', linestyle='--', label="No effect (0)")


plt.xlabel("Years Relative to Treatment")
plt.ylabel("Event-Study Coefficient")
plt.title("Event-Study: PM10 Before and After ZTL (Parallel Trends Check)")


legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markersize=8, label='Pre-treatment'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
           markersize=8, label='Post-treatment'),
    Line2D([0], [0], color='black', linestyle='--', label='Zero effect')
]
plt.legend(handles=legend_elements)

plt.grid(True)
plt.show()


# PRINT RESULTS
print("Event-study coefficients (relative to treatment year 2015):")
print(event_df[['t_rel','coef_adj','se','ci_low_adj','ci_high_adj']])