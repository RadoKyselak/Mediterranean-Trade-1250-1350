import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pointbiserialr
import statsmodels.api as sm
from pathlib import Path

ROUTES_FILE = "Trade Routes 1200-1300.csv"
INSTITUTIONS_FILE = "Medieval_Mediterranean_Institutional_Data_1250-1350.xlsx - Institutional Data.csv"
OUTPUT_DIR = Path("analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_medieval_analysis():
    print("Starting Medieval Institutional Analysis...")

    df_routes = pd.read_csv(ROUTES_FILE)
    df_routes = df_routes.groupby(['Source_City', 'Target_City'], as_index=False)['ROUTE_WEIGHT'].sum()

    df_inst = pd.read_csv(INSTITUTIONS_FILE).replace('?', np.nan)
    df_inst.columns = df_inst.columns.str.strip()

    inst_cols = ['Guilds', 'Port_Regulation', 'Comm_Court', 'Institutional_Autonomy', 'Population']
    for col in inst_cols:
        if col in df_inst.columns:
            df_inst[col] = pd.to_numeric(df_inst[col], errors='coerce')

    print("Building trade network...")
    G = nx.Graph()
    for _, row in df_routes.iterrows():
        strength = row['ROUTE_WEIGHT']
        distance = 1.0 / strength if strength > 0 else 1.0
        G.add_edge(row['Source_City'], row['Target_City'], weight=strength, distance=distance)

    centrality = nx.betweenness_centrality(G, weight='distance', normalized=True)
    df_centrality = pd.DataFrame(list(centrality.items()), columns=['City', 'Betweenness'])

    df_final = pd.merge(df_inst, df_centrality, on='City', how='inner')
    
    df_final.to_csv(OUTPUT_DIR / "merged_research_data.csv", index=False)

    print("Calculating statistics...")
    df_stats = df_final.dropna(subset=['Institutional_Autonomy', 'Betweenness'])
    
    rho, p_rho = spearmanr(df_stats['Institutional_Autonomy'], df_stats['Betweenness'])
    
    pb_list = []
    for col in ['Guilds', 'Port_Regulation', 'Comm_Court']:
        temp = df_final[[col, 'Betweenness']].dropna()
        if not temp.empty:
            r_pb, p_pb = pointbiserialr(temp[col], temp['Betweenness'])
            pb_list.append({'Variable': col, 'Correlation': r_pb, 'P-Value': p_pb})
    df_pb = pd.DataFrame(pb_list)

    X = sm.add_constant(df_stats['Institutional_Autonomy'])
    ols_res = sm.OLS(df_stats['Betweenness'], X).fit()

    print("Generating plots...")
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.regplot(data=df_stats, x='Institutional_Autonomy', y='Betweenness', 
                scatter_kws={'s': 80, 'alpha': 0.7}, line_kws={'color': "#d90202"})
    plt.title(f"Institutional Complexity vs City Status\n(Rho: {rho:.4f}, p: {p_rho:.4f}, n: {len(df_stats)})")
    plt.xlabel("Institutional Autonomy (Sum of Scores)")
    plt.ylabel("Betweenness Centrality (Network Node Score)")

    plt.subplot(1, 2, 2)
    heat_cols = ['Betweenness', 'Institutional_Autonomy', 'Guilds', 'Port_Regulation', 'Comm_Court']
    heat_cols = [c for c in heat_cols if c in df_final.columns]
    sns.heatmap(df_final[heat_cols].corr(method='spearman'), annot=True, cmap='viridis', center=0, fmt=".2f")
    plt.title("Institutional Component Spearman Matrix")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis_visuals.png", dpi=300)
    
    with open(OUTPUT_DIR / "regression_summary.txt", "w") as f:
        f.write(ols_res.summary().as_text())
    
    df_pb.to_csv(OUTPUT_DIR / "component_correlations.csv", index=False)

    print(f"\nDONE! Files saved in: {OUTPUT_DIR}/")
    print(f"Main Result: Spearman Rho = {rho:.4f} (p = {p_rho:.4f})")

if __name__ == "__main__":
    run_medieval_analysis()