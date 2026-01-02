
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

df = pd.read_csv("data/dataset_pm10.csv")

# LINE GRAPH 

# average
df_mean = df.groupby(['Anno', 'Treated'])['PM10'].mean().reset_index()

sns.set(style="whitegrid")
plt.figure(figsize=(12,6))

#post treatment shading
plt.axvspan(2015, 2020, color='gray', alpha=0.1)

# average 
lineplot = sns.lineplot(
    data=df_mean,
    x='Anno',
    y='PM10',
    hue='Treated',
    marker='o',
    palette={1:"#1f77b4", 0:"#ff7f0e"}
)

#treatment line
plt.axvline(x=2015, color='gray', linestyle='--')


plt.title("PM10 Trends: Treated vs Non-Treated Cities (2012-2018)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("PM10", fontsize=12)
plt.xticks(df['Anno'].unique())

legend_elements = [
    Line2D([0], [0], color="#1f77b4", marker='o', label='Treated'),
    Line2D([0], [0], color="#ff7f0e", marker='o', label='Non-Treated'),
    Patch(facecolor='gray', alpha=0.1, label='Post-Treatment')
]
plt.legend(handles=legend_elements, fontsize=11)

plt.show()
print("Average PM10 levels by Year and Treatment Status:")
print(df_mean)

# BARPLOT 

df_bar = df[df['Anno'].isin([2011, 2020])]

treated_cities = ['City1', 'City2']  
df_bar['treated'] = df_bar['Città'].apply(lambda x: 1 if x in treated_cities else 0)

df_bar['Anno_Treated'] = list(zip(df_bar['Anno'], df_bar['Treated']))

sns.set(style="whitegrid")

palette_colors = {
    (2011, 0): "#1f77b4",  # 2011, not treated
    (2011, 1): "#aec7e8",  # 2011, treated
    (2020, 0): "#ff7f0e",  # 2020, not treated
    (2020, 1): "#ffbb78"   # 2020, treated
}

plt.figure(figsize=(12,6))
sns.barplot(data=df_bar, x='Città', y='PM10', hue='Anno_Treated', palette=palette_colors)
plt.legend().remove()  

plt.title("PM10 Levels in 2011 vs 2020 by City", fontsize=14)
plt.xlabel(" ")
plt.ylabel("PM10", fontsize=12)
plt.show() 

# SCATTERPLOT

sns.set(style="whitegrid")
palette_colors = {1:"#1f77b4", 0:"#ff7f0e"}  

def scatter_year(year):
    df_year = df[df['Anno']==year]
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df_year,
        x='Città',
        y='PM10',
        hue='Treated',
        palette=palette_colors,
        s=150
    )
    plt.title(f"PM10 in {year}", fontsize=14)
    plt.xlabel("City", fontsize=12)
    plt.ylabel("PM10", fontsize=12)
    plt.xticks(rotation=45)
    

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor="#1f77b4", markersize=10, label='Treated'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor="#ff7f0e", markersize=10, label='Non-Treated')
    ]
    plt.legend(handles=legend_elements, fontsize=12)
    plt.tight_layout()
    plt.show()

scatter_year(2011)
scatter_year(2020)