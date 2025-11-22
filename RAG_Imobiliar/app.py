import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime
import json

# import project modules (retrieval, pricing, LLM explanation)
from retrieval import get_comparables
from pricing_model import compute_fair_price, load_comparables, evaluate_property
from explanation_module import generate_explanation_local

# State variables for Streamlit (persist between reruns)
# query_ran: used to detect if user pressed the button
# last_inputs: holds the last set of UI inputs
if "query_ran" not in st.session_state:
    st.session_state["query_ran"] = False
if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = {}

# Helper to assemble final JSON output for download
def build_final_output(target_title, listed_price, target_sqm, comparables, estimation):
    output = {
        "decision_target_id": f"auto_{int(datetime.utcnow().timestamp())}",  # unique ID for the target property
        "title": target_title,
        "listed_price_eur": int(listed_price) if listed_price is not None else None,
        "fair_price_eur": estimation["fair_price"],
        "fair_range_eur": [
            estimation["confidence_interval"]["lower"],
            estimation["confidence_interval"]["upper"]
        ],
        "label": estimation["verdict"],
        "comparables_used": comparables  # list of k comparables with scores
    }
    return output

# UI LAYOUT
st.set_page_config(layout="wide", page_title="RAG Imobiliar — Demo")
st.title("RAG Imobiliar — Estimare pret corect (demo)")

# Sidebar = Filter panel + target property info
with st.sidebar:
    st.header("Filtre")

    # Retrieval filters used to build natural-language query
    property_type = st.selectbox("Tip proprietate", options=["any", "apartment", "house", "land"])
    neighborhood = st.text_input("Cartier (ex: Titan)", value="")
    min_size = st.number_input("Suprafata minima (mp)", min_value=0, value=40)
    max_budget = st.number_input("Buget maxim (EUR)", min_value=0, value=120000)
    k = st.slider("Numar comparabile (k)", 3, 20, 10)

    st.markdown("---")
    st.markdown("Input pentru proprietatea analizata (optional):")

    # Inputs for the target property that we estimate the fair price for
    title_input = st.text_input("Titlu proprietate", value="Proprietate exemplu")
    listed_price_input = st.number_input("Pret listat (EUR)", min_value=0, value=60000)
    size_input = st.number_input("Suprafata (mp)", min_value=1, value=54)

    # Button that triggers full pipeline (retrieval → pricing → explanation → map)
    run_button = st.button("Calculeaza preț corect")

# Handle button click
if run_button:
    # Save inputs in session state so page does not reset on rerun
    st.session_state["query_ran"] = True
    st.session_state["last_inputs"] = {
        "property_type": property_type,
        "neighborhood": neighborhood,
        "min_size": min_size,
        "max_budget": max_budget,
        "k": k,
        "title_input": title_input,
        "listed_price_input": listed_price_input,
        "size_input": size_input
    }

# Main execution block
if st.session_state["query_ran"]:
    inp = st.session_state["last_inputs"]
    # Build natural-language query
    q_parts = []
    if inp["property_type"] != "any":
        q_parts.append(inp["property_type"])
    if inp["neighborhood"]:
        q_parts.append(inp["neighborhood"])
    q_parts.append(f"{int(inp['size_input'])} mp")
    if inp["listed_price_input"]:
        q_parts.append(f"buget {int(inp['listed_price_input'])} euro")

    query_text = " ".join(q_parts)

    # Show query for debugging / transparency
    st.info(f"Rulez: {query_text}")

    # Retrieves top-k comparables after vector search + ranking
    comparables = get_comparables(query_text, k=inp["k"])

    # Show comparables table
    df_comps = pd.DataFrame(comparables)
    st.subheader("Comparabile (dupa reranking)")
    st.dataframe(df_comps[[
        "id","property_type","neighborhood",
        "price_eur","size_sqm","price_per_sqm","final_score"
    ]])

    # Prepare comparable entries with fields needed for scoring
    comps_for_pricing = []
    for c in comparables:
        comps_for_pricing.append({
            "id": c["id"],
            "final_score": c["final_score"],          # combined score after ranking
            "price_per_sqm": c["price_per_sqm"],      # numeric value used in weighted average
            "size_sqm": c["size_sqm"],
            "price_eur": c["price_eur"],
            "neighborhood": c["neighborhood"],
            "similarity": c.get("similarity", None)    # raw embedding similarity (optional)
        })

    # Compute fair price estimation (weighted PPSQM)
    estimation = compute_fair_price(
        comps_for_pricing,
        target_price=inp["listed_price_input"],
        target_sqm=inp["size_input"]
    )

    # Pricing UI Section
    st.subheader("Estimare preț corect")
    col1, col2, col3 = st.columns([1,1,2])
    col1.metric("Preț corect (EUR)", f"{estimation['fair_price']:.0f}")
    col2.metric("Preț corect (€/mp)", f"{estimation['fair_ppsqm']:.0f}")
    col3.markdown(
        f"**Interval ±5%:** {estimation['confidence_interval']['lower']:.0f} — {estimation['confidence_interval']['upper']:.0f}"
    )
    st.markdown(f"**Verdict:** `{estimation['verdict']}`")

    # EXPLANATION (LLM or rule-based)
    explanation = generate_explanation_local({
        "estimation": {
            "fair_price": estimation['fair_price'],
            "fair_ppsqm": estimation['fair_ppsqm'],
            "confidence_interval": estimation['confidence_interval'],
            "verdict": estimation['verdict'],
            "target_sqm": inp['size_input']
        },
        "comparables_used": comps_for_pricing,
        "title": inp["title_input"],
        "listed_price_eur": inp["listed_price_input"]
    })

    st.subheader("Explicație")
    st.write(explanation["explanation_text"])
    st.caption(explanation["disclaimer"])

    # MAP SECTION (Folium)
    st.subheader("Hartă")

    # Load the full dataset to extract coords of comparables
    df_all = pd.read_csv("properties_clean.csv")
    df_coords = df_all.set_index("id").loc[[c["id"] for c in comparables]].reset_index()

    # Determine map center based on average coordinates of comparables
    if not df_coords.empty:
        center_lat = df_coords["lat"].mean()
        center_lon = df_coords["lon"].mean()
    else:
        # fallback: Bucharest center
        center_lat, center_lon = 44.4268, 26.1025

    # Create Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Compute quantiles for price-per-sqm coloring
    pps = df_coords["price_per_sqm"].fillna(0)
    if not pps.empty:
        q1 = pps.quantile(0.25)
        q2 = pps.quantile(0.5)
        q3 = pps.quantile(0.75)
    else:
        q1 = q2 = q3 = 0

    # Function to color markers by price segment
    def color_by_pps(v):
        if v <= q1: return "green"
        if v <= q2: return "lightgreen"
        if v <= q3: return "orange"
        return "red"

    # Add markers to map
    for _, r in df_coords.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6,
            color=color_by_pps(r["price_per_sqm"]),
            fill=True,
            fill_opacity=0.8,
            popup=f"ID:{int(r['id'])} {int(r['price_eur'])} EUR — {int(r['price_per_sqm'])} €/mp"
        ).add_to(m)

    st_folium(m, width=800)

    # JSON export
    final_json = build_final_output(
        inp["title_input"],
        inp["listed_price_input"],
        inp["size_input"],
        comps_for_pricing,
        {
            "fair_price": estimation['fair_price'],
            "confidence_interval": estimation['confidence_interval'],
            "verdict": estimation['verdict']
        }
    )

    st.subheader("Export")
    st.download_button(
        "Descarcă rezultatul JSON",
        data=json.dumps(final_json, ensure_ascii=False, indent=2),
        file_name="final_output.json",
        mime="application/json"
    )

else:
    # Initial state before pressing the button
    st.info("Completeaza filtrele in sidebar si apasa 'Calculeaza preț corect'.")
