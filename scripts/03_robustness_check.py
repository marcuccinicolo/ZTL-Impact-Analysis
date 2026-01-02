import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv("data/dataset_pm10.csv")
df['Anno'] = df['Anno'].astype(int)
df['Città'] = df['Città'].astype(str)
df['Treated'] = df['Treated'].astype(int)
df = df.dropna(subset=['PM10','Treated','Anno'])


# leave-one-out

loo_cities = ["Torino", "Milano", "Firenze", "La Spezia", "Pescara"]
policy_year = 2015

for loo_city in loo_cities:
    df_loo = df[df['Città'] != loo_city].copy()
    
    # relative year
  
    df_loo['t_rel'] = df_loo['Anno'] - policy_year
    df_loo['t_rel'] = df_loo['t_rel'].astype('category')
    df_loo['t_rel'] = df_loo['t_rel'].cat.reorder_categories(sorted(df_loo['t_rel'].cat.categories), ordered=True)
    

    # Event-study regression

    model = smf.ols(
        "PM10 ~ Treated:t_rel + C(Città) + C(Anno)",
        data=df_loo
    ).fit(cov_type="cluster", cov_kwds={'groups': df_loo['Città']})
    
    # coefficients and CI
  
    event_terms = [c for c in model.params.index if "Treated:" in c]
    event_df = pd.DataFrame({
        "t_rel": [int(term.split("t_rel")[1].strip("[]")) for term in event_terms],
        "coef": [model.params[term] for term in event_terms],
        "se": [model.bse[term] for term in event_terms]
    })
    event_df = event_df.sort_values("t_rel")
    event_df["ci_low"] = event_df["coef"] - 1.96 * event_df["se"]
    event_df["ci_high"] = event_df["coef"] + 1.96 * event_df["se"]
    
    pre_treat_mask = event_df['t_rel'] < 0
    pre_mean = event_df.loc[pre_treat_mask, 'coef'].mean()
    event_df['coef_adj'] = event_df['coef'] - pre_mean
    event_df['ci_low_adj'] = event_df['ci_low'] - pre_mean
    event_df['ci_high_adj'] = event_df['ci_high'] - pre_mean
    

    # Plot

    plt.figure(figsize=(10,6))
    colors = ["blue" if t < 0 else "orange" for t in event_df["t_rel"]]

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
    plt.title(f"Event-Study Leave-One-Out: PM10 without {loo_city}")

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
    
    # print
    print(f"Event-study coefficients without {loo_city} (relative to treatment year {policy_year}):")
    print(event_df[['t_rel','coef_adj','se','ci_low_adj','ci_high_adj']])


# DYNAMIC PLACEBO 

non_treated = df[df['Treated'] == 0]['Città'].unique()
placebo_cities = np.random.choice(non_treated, size=5, replace=False)

df_placebo = df.copy()
df_placebo['Treated_placebo'] = df_placebo['Città'].apply(lambda x: 1 if x in placebo_cities else 0)

# relatve year
policy_year = 2015
df_placebo['t_rel'] = df_placebo['Anno'] - policy_year
df_placebo['t_rel'] = df_placebo['t_rel'].astype('category')
df_placebo['t_rel'] = df_placebo['t_rel'].cat.reorder_categories(sorted(df_placebo['t_rel'].cat.categories), ordered=True)


# Event-study regression placebo

model = smf.ols(
    "PM10 ~ Treated_placebo:t_rel + C(Città) + C(Anno)",
    data=df_placebo
).fit(cov_type="cluster", cov_kwds={'groups': df_placebo['Città']})

# coefficients and CI
event_terms = [c for c in model.params.index if "Treated_placebo:" in c]
event_df = pd.DataFrame({
    "t_rel": [int(term.split("t_rel")[1].strip("[]")) for term in event_terms],
    "coef": [model.params[term] for term in event_terms],
    "se": [model.bse[term] for term in event_terms]
})
event_df = event_df.sort_values("t_rel")

pre_mean = event_df.loc[event_df['t_rel'] < 0, 'coef'].mean()
event_df['coef_adj'] = event_df['coef'] - pre_mean
event_df['ci_low'] = event_df['coef_adj'] - 1.96*event_df['se']
event_df['ci_high'] = event_df['coef_adj'] + 1.96*event_df['se']

# print
print("\nPlacebo Event-Study Coefficients (5 random non-treated cities):")
print(event_df[['t_rel','coef_adj','se','ci_low','ci_high']])

# graph
plt.figure(figsize=(10,6))
colors = ["blue" if t < 0 else "orange" for t in event_df["t_rel"]]

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
plt.title("Placebo Event-Study: 5 Non-Treated Cities")

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Pre-treatment'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Post-treatment'),
    Line2D([0], [0], color='black', linestyle='--', label='Zero effect')
]
plt.legend(handles=legend_elements)
plt.grid(True)
plt.show()


# PLACEBO RANDOM CITIES FROM FULL SAMPLE

all_cities = df['Città'].unique()

# 5 random cities
np.random.seed(42)   
placebo_cities_all = np.random.choice(all_cities, size=5, replace=False)

print("\nCittà placebo (random full sample):", placebo_cities_all)

df_placebo_all = df.copy()
df_placebo_all['Treated_placebo_all'] = df_placebo_all['Città'].apply(
    lambda x: 1 if x in placebo_cities_all else 0
)


# relative year

df_placebo_all['t_rel'] = df_placebo_all['Anno'] - policy_year
df_placebo_all['t_rel'] = df_placebo_all['t_rel'].astype('category')
df_placebo_all['t_rel'] = df_placebo_all['t_rel'].cat.reorder_categories(
    sorted(df_placebo_all['t_rel'].cat.categories),
    ordered=True
)


# Event-study placebo

model_all = smf.ols(
    "PM10 ~ Treated_placebo_all:t_rel + C(Città) + C(Anno)",
    data=df_placebo_all
).fit(cov_type="cluster", cov_kwds={'groups': df_placebo_all['Città']})

# coefficients and CI
event_terms_all = [c for c in model_all.params.index if "Treated_placebo_all:" in c]

event_df_all = pd.DataFrame({
    "t_rel": [int(t.split("t_rel")[1].strip("[]")) for t in event_terms_all],
    "coef": [model_all.params[t] for t in event_terms_all],
    "se": [model_all.bse[t] for t in event_terms_all]
}).sort_values("t_rel")

pre_mean = event_df_all.loc[event_df_all['t_rel'] < 0, 'coef'].mean()
event_df_all['coef_adj'] = event_df_all['coef'] - pre_mean
event_df_all['ci_low'] = event_df_all['coef_adj'] - 1.96 * event_df_all['se']
event_df_all['ci_high'] = event_df_all['coef_adj'] + 1.96 * event_df_all['se']

#print

print("\nPlacebo Event-Study (5 random cities from full sample):")
print(event_df_all[['t_rel','coef_adj','se','ci_low','ci_high']])


# Plot

plt.figure(figsize=(10,6))
colors = ["blue" if t < 0 else "orange" for t in event_df_all["t_rel"]]

for i, row in event_df_all.iterrows():
    plt.errorbar(row["t_rel"], row["coef_adj"],
                 yerr=1.96*row["se"],
                 fmt='o',
                 color=colors[i],
                 ecolor=colors[i],
                 elinewidth=2,
                 capsize=4)

plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Years Relative to Treatment")
plt.ylabel("Event-Study Coefficient")
plt.title("Placebo Event-Study: 5 Random Cities (Any Type)")
plt.grid(True)
plt.show()