import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    path = "data_output/fastf1_races.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df
    return pd.DataFrame()

def main():
    st.set_page_config(page_title="KRONECTOR Dashboard", layout="wide")
    
    st.title("🏎️ KRONECTOR - F1 Analytics Dashboard")
    st.markdown("Analyze model drift and explore historical F1 era comparisons.")
    
    tab1, tab2 = st.tabs(["📊 Drift Report", "🏎️ Era Comparison Analytics"])
    
    with tab1:
        st.header("Evidently AI Drift Report")
        report_path = "ui/drift_report.html"
        
        if os.path.exists(report_path):
            st.success("Latest Drift Report Found!")
            with open(report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=1000, scrolling=True)
        else:
            st.warning("No drift report found. Please run the auto-retraining pipeline to generate one.")
            st.info("Run `python -m scripts.auto_retrain_pipeline` in your terminal.")

    with tab2:
        st.header("F1 Era Comparison")
        st.markdown("Compare car performance (lap times & sectors) across the Hybrid, Ground Effect, and Agile eras.")
        
        df = load_data()
        
        if df.empty:
            st.warning("No dataset found. Please run the pipeline first.")
            return
            
        # Filter out rows ONLY if BOTH lap times and sector times are missing
        df_clean = df.dropna(subset=["avg_lap_time_practice", "sector_1_time"], how="all")
        
        circuits = sorted(df_clean["circuit_id"].dropna().unique())
        selected_circuit = st.selectbox("Select a Circuit", circuits)
        
        circuit_df = df_clean[df_clean["circuit_id"] == selected_circuit]
        
        if circuit_df.empty:
            st.info("No lap time data available for this circuit.")
            return
            
        # Aggregate data by season and regulation_era
        agg_cols = ["avg_lap_time_practice", "sector_1_time", "sector_2_time", "sector_3_time"]
        era_agg = circuit_df.groupby(["season", "regulation_era"])[agg_cols].mean().reset_index()
        era_agg = era_agg.sort_values("season")
        
        # Color mapping for eras
        color_map = {
            "hybrid_era": "#3498db",        # Blue
            "ground_effect_era": "#e74c3c", # Red
            "agile_era": "#2ecc71"          # Green
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_lap = px.bar(
                era_agg, 
                x="season", 
                y="avg_lap_time_practice", 
                color="regulation_era",
                color_discrete_map=color_map,
                title=f"Average Practice Lap Time (Seconds) - {selected_circuit}",
                labels={"avg_lap_time_practice": "Lap Time (s)", "season": "Season", "regulation_era": "Era"}
            )
            # Adjust y-axis to emphasize differences
            min_time = era_agg["avg_lap_time_practice"].min()
            max_time = era_agg["avg_lap_time_practice"].max()
            fig_lap.update_yaxes(range=[min_time * 0.95, max_time * 1.05])
            st.plotly_chart(fig_lap, use_container_width=True)
            
        with col2:
            fig_s1 = px.bar(
                era_agg.dropna(subset=["sector_1_time"]), 
                x="season", 
                y="sector_1_time", 
                color="regulation_era",
                color_discrete_map=color_map,
                title=f"Average Sector 1 Time (Seconds) - {selected_circuit}",
                labels={"sector_1_time": "Sector 1 Time (s)", "season": "Season", "regulation_era": "Era"}
            )
            if not era_agg["sector_1_time"].isna().all():
                min_s1 = era_agg["sector_1_time"].min()
                max_s1 = era_agg["sector_1_time"].max()
                fig_s1.update_yaxes(range=[min_s1 * 0.9, max_s1 * 1.1])
            st.plotly_chart(fig_s1, use_container_width=True)

        st.markdown("### Era Insights")
        fastest_row = era_agg.loc[era_agg["avg_lap_time_practice"].idxmin()]
        slowest_row = era_agg.loc[era_agg["avg_lap_time_practice"].idxmax()]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Fastest Season", f"{int(fastest_row['season'])} ({fastest_row['regulation_era']})", f"{fastest_row['avg_lap_time_practice']:.2f}s", delta_color="inverse")
        m2.metric("Slowest Season", f"{int(slowest_row['season'])} ({slowest_row['regulation_era']})", f"{slowest_row['avg_lap_time_practice']:.2f}s", delta_color="normal")
        m3.metric("Max Pace Difference", f"{slowest_row['avg_lap_time_practice'] - fastest_row['avg_lap_time_practice']:.2f}s")

if __name__ == "__main__":
    main()
