# Impact of Urban Traffic Restrictions (ZTL) on PM10 Levels

This project evaluates the causal impact of **Limited Traffic Zones (ZTL)** introduced in 2015 on air quality in major Italian cities (Milan, Turin, Florence, etc.).

## 🔬 Methodology: Difference-in-Differences (DiD)
The analysis uses a **Difference-in-Differences** econometric approach to isolate the effect of the policy:
- **Treatment Group:** Cities that implemented ZTL in 2015.
- **Control Group:** Cities with similar characteristics but no ZTL in that period.
- **Fixed Effects:** Inclusion of City and Year fixed effects to control for time-invariant characteristics and common shocks.



## 📈 Key Results
- **Estimated effect:** PM10 in the ZTL cities fell by **4.94 µg/m³ more** than in the
  control cities after 2015 (se 1.04, p < 0.001, 95% CI [-6.98, -2.91]), with city and
  year fixed effects and standard errors clustered by city.
- **Parallel Trends:** Validated through an Event-Study design (see `scripts/02_did_model.py`).
  Pre-2015 coefficients straddle zero; the gap opens from the year of introduction.
- **Robustness:** Includes "Leave-one-out" analysis and "Placebo tests" to confirm the validity of the results.

### What this does not identify
- **15 cities is a small number of clusters**, so the clustered standard errors deserve
  more caution than their size suggests.
- 2015 was not a year in which the treated cities changed *only* their traffic zone. The
  estimate belongs to the package of measures a city adopting a ZTL tends to adopt, not
  to the ZTL alone.

## ⚙️ Execution
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main model: `python scripts/02_did_model.py`
