# Impact of Urban Traffic Restrictions (ZTL) on PM10 Levels

This project evaluates the causal impact of **Limited Traffic Zones (ZTL)** introduced in 2015 on air quality in major Italian cities (Milan, Turin, Florence, etc.).

## 🔬 Methodology: Difference-in-Differences (DiD)
The analysis uses a **Difference-in-Differences** econometric approach to isolate the effect of the policy:
- **Treatment Group:** Cities that implemented ZTL in 2015.
- **Control Group:** Cities with similar characteristics but no ZTL in that period.
- **Fixed Effects:** Inclusion of City and Year fixed effects to control for time-invariant characteristics and common shocks.



## 📈 Key Results
- **Parallel Trends:** Validated through an Event-Study design (see `SCRIPTS/02_did_model.py`).
- **Robustness:** Includes "Leave-one-out" analysis and "Placebo tests" to confirm the validity of the results.

## ⚙️ Execution
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main model: `python SCRIPTS/02_did_model.py`
