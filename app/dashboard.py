"""Streamlit dashboard over persisted outputs; no model fitting on app startup."""
import streamlit as st
import pandas as pd
from data_loader import load_dashboard_data, artifact_path

st.set_page_config(page_title="Context-Aware Inventory Research",page_icon="📦",layout="wide")
data=load_dashboard_data()
st.title("Context-Aware Stochastic Inventory Optimization")
st.caption("Research dashboard · saved results only · no optimization is run when this app opens")
section=st.sidebar.radio("Explore",["Project overview","Demand analysis","Optimization results","Simulation results","Robustness analysis","Research conclusions"])

def notice_if_missing(keys):
    missing=[k for k in keys if data.get(k) is None]
    if missing: st.info("Some saved outputs are not available yet: "+", ".join(missing)+". Run the corresponding project modules to populate them.")

if section=="Project overview":
    st.header("Project overview")
    st.write("Research question: does contextual stochastic inventory optimization improve realized cost and service relative to non-contextual optimization and a historical order-up-to policy?")
    st.markdown("**Pipeline:** validated source data → contextual demand distributions → stochastic order decisions → historical and repeated simulation → paired robustness analysis.")
    notice_if_missing(["dataset_summary"])
    summary_path=load_dashboard_data().get("dataset_summary")
    # Dataset summary is JSON, loaded separately and gracefully.
    summary=data.get("dataset_summary")
    if summary:
        try:
            cols=st.columns(4)
            for col,key in zip(cols,["rows","products","categories","suppliers"]): col.metric(key.replace("_"," ").title(),f"{summary[key]:,}")
            st.caption(f"Dataset dates: {summary['date_start']} to {summary['date_end']}. The dataset is synthetic.")
        except (ValueError,KeyError): st.info("Dataset summary JSON could not be read.")
    st.markdown("**Method:** daily product demand is modeled as a count distribution; inventory decisions minimize expected purchase, ordering, holding and lost-sales cost. Historical holdout and bootstrap simulation are shown separately.")

elif section=="Demand analysis":
    st.header("Demand analysis")
    for title,name in [("Historical demand trend","daily_demand_trend"),("Demand distribution","demand_distribution"),("Demand by category","demand_by_category"),("Monthly demand","monthly_demand"),("Seasonal demand","seasonal_demand"),("Weekly pattern","weekly_demand"),("Festival demand","festival_demand"),("Supplier context","supplier_profile")]:
        path=data["figures"][name]
        if path.is_file(): st.subheader(title); st.image(str(path),use_container_width=True)
    st.subheader("Category summary")
    if data["category_demand"] is not None: st.dataframe(data["category_demand"],use_container_width=True)
    else: st.info("Category demand summary is not available.")
    st.subheader("Product summary")
    if data["product_demand"] is not None: st.dataframe(data["product_demand"],use_container_width=True)
    else: st.info("Product demand summary is not available.")

elif section=="Optimization results":
    st.header("Optimization results · Module 3 scenario estimates")
    frame=data["optimization"]
    if frame is None: st.info("Module 3 optimization results are not available.")
    else:
        st.dataframe(frame,use_container_width=True)
        if {"model","expected_total_cost"}.issubset(frame.columns): st.bar_chart(frame.set_index("model")[["expected_total_cost"]])
    for label,key in [("Contextual product orders","contextual_orders"),("Non-contextual product orders","noncontextual_orders")]:
        st.subheader(label)
        if data[key] is not None: st.dataframe(data[key],use_container_width=True)
        else: st.info(f"{label} are not available.")

elif section=="Simulation results":
    st.header("Historical holdout simulation · Module 4")
    frame=data["historical_simulation"]
    if frame is None: st.info("Module 4 simulation results are not available.")
    else:
        st.dataframe(frame,use_container_width=True)
        metric=st.selectbox("Compare metric",[c for c in ["total_inventory_cost_inr","fill_rate","total_lost_sales_units","stockout_frequency","average_on_hand_inventory_units"] if c in frame.columns])
        st.bar_chart(frame.set_index("policy")[[metric]])

elif section=="Robustness analysis":
    st.header("Repeated stochastic simulation · Module 5")
    notice_if_missing(["repeated_summary","paired_comparison","sensitivity"])
    if data["repeated_summary"] is not None:
        metric=st.selectbox("Repeated metric",sorted(data["repeated_summary"].metric.unique()))
        view=data["repeated_summary"].query("metric == @metric").set_index("policy")
        st.dataframe(view,use_container_width=True)
        if "mean" in view.columns: st.bar_chart(view[["mean"]])
    if data["repeated_cost_breakdown"] is not None:
        st.subheader("Repeated cost breakdown")
        st.dataframe(data["repeated_cost_breakdown"],use_container_width=True)
    if data["paired_comparison"] is not None:
        st.subheader("Paired contextual vs non-contextual comparison")
        st.dataframe(data["paired_comparison"],use_container_width=True)
    if data["sensitivity"] is not None:
        st.subheader("Frozen-plan sensitivity analysis")
        st.dataframe(data["sensitivity"],use_container_width=True)
    for title,name in [("Repeated policy cost","repeated_policy_cost"),("Repeated fill rate","repeated_policy_fill_rate"),("Sensitivity cost","sensitivity_cost")]:
        path=data["report_figures"][name]
        if path.is_file(): st.subheader(title); st.image(str(path),use_container_width=True)

else:
    st.header("Research conclusions")
    if data["research_report"]: st.markdown(data["research_report"])
    else: st.info("Final research report is not available yet. Run Module 5 after Modules 1–4.")
