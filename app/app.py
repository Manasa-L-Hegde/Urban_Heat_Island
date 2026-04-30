"""Streamlit dashboard for the Urban Heat Island Mapping app.

This version uses session state to switch cleanly between:
- map page
- chatbot page

The green floating button navigates to the chatbot page,
and the back button returns to the map page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit_folium import st_folium
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'streamlit_folium'. Install it with `pip install streamlit-folium` "
        "or run `pip install -r requirements.txt` from the repository root."
    ) from exc

# Add parent directory to path so src imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fusion import fuse_temperatures
from src.heat_index import build_heat_index
from src.heat_risk_advisor import ask_ai, build_context, generate_green_cover_recommendations
from src.map_builder import build_map

DATA_PATH = ROOT / "data" / "ward_data.csv"

st.set_page_config(
    page_title="Urban Heat Island Mapping",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables for page navigation and chatbot state
if "page" not in st.session_state:
    st.session_state.page = "map"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""


def set_page(new_page: str) -> None:
    st.session_state.page = new_page


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    raise KeyError(f"None of these columns were found: {candidates}")


def _risk_badge(category: str) -> str:
    normalized = str(category).strip().lower()
    if "high" in normalized:
        return "🔴 High"
    if "medium" in normalized:
        return "🟡 Medium"
    if "low" in normalized:
        return "🟢 Low"
    return str(category)


# Page styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0F1724;
            color: #F8FAFC;
        }
        .hero-card,
        .section-card,
        .mini-card,
        .top-tier-card,
        .chat-focus {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.20);
        }
        .hero-title {
            color: #8B5CF6;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }
        .hero-subtitle {
            color: #94A3B8;
            font-size: 1rem;
            line-height: 1.8;
            margin-top: 0;
        }
        .section-title {
            color: #E2E8F0;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .metric-label {
            color: #94A3B8;
            font-size: 0.85rem;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: #F8FAFC;
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
        }
        .badge-high,
        .badge-medium,
        .badge-low {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 14px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .badge-high { background: rgba(248, 113, 113, 0.18); color: #FECACA; }
        .badge-medium { background: rgba(250, 204, 21, 0.18); color: #FDE047; }
        .badge-low { background: rgba(52, 211, 153, 0.18); color: #A7F3D0; }
        .top-tier-card {
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(56, 189, 248, 0.15));
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 18px;
            min-height: 160px;
        }
        .top-tier-title {
            color: #C4B5FD;
            font-size: 1rem;
            margin-bottom: 0.75rem;
            font-weight: 700;
        }
        .top-tier-value {
            color: #F8FAFC;
            font-size: 1.75rem;
            font-weight: 800;
            margin: 0;
        }
        .sidebar .css-1d391kg { padding-top: 0; }
        .sidebar .stMarkdown h3 { color: #F8FAFC; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error("Missing dataset: data/ward_data.csv. Run src/data_generation.py first.")
    st.stop()

raw_df = pd.read_csv(DATA_PATH)
processed_df = build_heat_index(fuse_temperatures(raw_df))
processed_df = generate_green_cover_recommendations(processed_df)
green_cover_col = _find_column(processed_df, ["Green Cover", "Green Cover (%)", "Green Cover %"])

if st.session_state.page == "map":
    with st.container():
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-title">🌡️ Urban Heat Island Dashboard</div>
                <p class="hero-subtitle">Modern ward-level heat risk visualization with clean cards, badges, and filtered data presentation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown(
            """
            <div style="background: #1a1f35; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <h3 style="color: #8B5CF6; margin-top: 0; font-size: 1.2rem;">🎛️ Filter & Sort</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        risk_filter = st.multiselect(
            "Risk Level",
            options=sorted(processed_df["Vulnerability_Category"].unique()),
            default=sorted(processed_df["Vulnerability_Category"].unique()),
            help="Choose which risk categories to display.",
        )
        
        ward_search = st.selectbox(
            "🔍 Search Ward",
            options=["All Wards"] + sorted(processed_df["Ward Name"].unique().tolist()),
            index=0,
            help="Select a specific ward to view details.",
        )
        
        sort_option = st.selectbox(
            "Sort by",
            options=["Heat_Index", "Final_Temperature", "Vulnerability_Index"],
            index=0,
            help="Sort the ward list by the selected metric.",
        )
        st.markdown("---")
        st.markdown(
            """
            <div style="background: #1a1f35; border-radius: 12px; padding: 14px; margin-top: 16px;">
                <h4 style="color: #F8FAFC; margin-top: 0; font-size: 1rem;">🔥 Risk Legend</h4>
                <p style="color: #FECACA; margin: 8px 0;"><strong>🔴 High Risk</strong><br/>Urgent attention needed</p>
                <p style="color: #FDE047; margin: 8px 0;"><strong>🟡 Medium Risk</strong><br/>Monitor closely</p>
                <p style="color: #A7F3D0; margin: 8px 0;"><strong>🟢 Low Risk</strong><br/>Stable wards</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    filtered_df = processed_df[processed_df["Vulnerability_Category"].isin(risk_filter)]
    
    if ward_search != "All Wards":
        filtered_df = filtered_df[filtered_df["Ward Name"] == ward_search]
    
    filtered_df = filtered_df.sort_values(sort_option, ascending=False).reset_index(drop=True)

    if filtered_df.empty:
        st.warning("No wards match the selected filters. Adjust risk levels or sort options to see results.")

    avg_temp = filtered_df["Final_Temperature"].mean() if not filtered_df.empty else 0.0
    avg_heat_index = filtered_df["Heat_Index"].mean() if not filtered_df.empty else 0.0
    high_risk_count = int(
        filtered_df[filtered_df["Vulnerability_Category"].str.lower().str.contains("high")].shape[0]
    )
    top5 = filtered_df.sort_values("Heat_Index", ascending=False).head(5)

    with st.container():
        card1, card2, card3 = st.columns(3)
        card1.markdown(
            f"""
            <div class="mini-card">
                <p class="metric-label">Average Temperature</p>
                <p class="metric-value">{avg_temp:.1f}°C</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card2.markdown(
            f"""
            <div class="mini-card">
                <p class="metric-label">Avg Heat Index</p>
                <p class="metric-value">{avg_heat_index:.1f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card3.markdown(
            f"""
            <div class="mini-card">
                <p class="metric-label">High Risk Wards</p>
                <p class="metric-value">{high_risk_count}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    with st.container():
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🔥 Top 5 Hottest Wards</div>
                <p style="color:#94A3B8; margin-top:0;">The wards with the highest heat index are shown below for fast insight.</p>
            """,
            unsafe_allow_html=True,
        )

        if top5.empty:
            st.info("No top wards available for the current filter selection.")
        else:
            top_columns = st.columns(min(5, len(top5)))
            for idx, (_, row) in enumerate(top5.reset_index(drop=True).iterrows()):
                top_columns[idx].markdown(
                    f"""
                    <div class="top-tier-card">
                        <p class="top-tier-title">{row['Ward Name']}</p>
                        <p class="top-tier-value">{row['Heat_Index']:.1f}</p>
                        <p class="badge-{row['Vulnerability_Category'].strip().lower()}">{_risk_badge(row['Vulnerability_Category'])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    with st.container():
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">📊 Ward-Level Data</div>
                <p style="color:#94A3B8; margin-top:0;">Browse the filtered ward dataset with heat index and vulnerability highlights.</p>
            """,
            unsafe_allow_html=True,
        )

        display_df = filtered_df.copy()
        display_df.insert(
            display_df.columns.get_loc("Vulnerability_Category") + 1,
            "Risk Badge",
            display_df["Vulnerability_Category"].apply(_risk_badge),
        )

        columns = [
            "Ward Name",
            "Final_Temperature",
            "Heat_Index",
            "Vulnerability_Index",
            "Vulnerability_Category",
            "Risk Badge",
            green_cover_col,
        ]
        if "Green_Cover_Recommendation" in display_df.columns:
            columns.append("Green_Cover_Recommendation")
        columns = [col for col in columns if col in display_df.columns]
        display_df = display_df[columns]

        display_df = display_df.round(2)
        st.dataframe(display_df, width='stretch', height=500)

    st.markdown("---")

    with st.container():
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">� Heat Index Distribution</div>
                <p style="color:#94A3B8; margin-top:0;">Visual representation of heat index across the top 15 filtered wards.</p>
            """,
            unsafe_allow_html=True,
        )
        import altair as alt
        if not filtered_df.empty:
            chart_data = filtered_df.nlargest(15, "Heat_Index")[["Ward Name", "Heat_Index"]].reset_index(drop=True)
            chart = alt.Chart(chart_data).mark_bar(color="#8B5CF6").encode(
                x=alt.X("Heat_Index:Q", title="Heat Index", scale=alt.Scale(zero=False)),
                y=alt.Y("Ward Name:N", title="Ward", sort="-x"),
                tooltip=["Ward Name:N", alt.Tooltip("Heat_Index:Q", format=".2f")]
            ).properties(
                width=700,
                height=350
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data available for the chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    with st.container():
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🌡️ Temperature vs Vulnerability</div>
                <p style="color:#94A3B8; margin-top:0;">Relationship between temperature and vulnerability index across wards.</p>
            """,
            unsafe_allow_html=True,
        )
        if not filtered_df.empty:
            scatter_data = filtered_df[["Ward Name", "Final_Temperature", "Vulnerability_Index", "Vulnerability_Category"]].reset_index(drop=True)
            scatter = alt.Chart(scatter_data).mark_circle(size=80).encode(
                x=alt.X("Final_Temperature:Q", title="Temperature (°C)"),
                y=alt.Y("Vulnerability_Index:Q", title="Vulnerability Index"),
                color=alt.Color("Vulnerability_Category:N", title="Risk Level"),
                tooltip=["Ward Name:N", alt.Tooltip("Final_Temperature:Q", format=".1f"), alt.Tooltip("Vulnerability_Index:Q", format=".2f")]
            ).properties(
                width=700,
                height=350
            )
            st.altair_chart(scatter, use_container_width=True)
        else:
            st.info("No data available for the chart.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    with st.container():
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">🗺️ Interactive Heat Map</div>
                <p style="color:#94A3B8; margin-top:0;">Explore ward heat distribution across the city. Hover over zones to see detailed risk metrics.</p>
                <hr style="border-color: rgba(148, 163, 184, 0.2); margin: 12px 0;">
            """,
            unsafe_allow_html=True,
        )

        if filtered_df.empty:
            st.warning("No map data available for the selected filters.")
        else:
            try:
                m = build_map(filtered_df, selected_ward=filtered_df["Ward Name"].iloc[0])
                st.markdown(
                    """
                    <div style="border-radius: 12px; overflow: hidden; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.22);">
                    """,
                    unsafe_allow_html=True,
                )
                st_folium(m, width=None, height=580, returned_objects=[])
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Map rendering error: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.button("Open Chatbot", key="hidden_chatbot", on_click=set_page, args=("chatbot",), help="Open chatbot page")
    st.markdown(
        """
        <style>
            button[key="hidden_chatbot"] { display:none !important; }
        </style>
        <button id="chatbot-float-btn" title="Open Chatbot">🤖</button>
        <script>
            const btn = document.getElementById('chatbot-float-btn');
            btn.onclick = () => {
                const hidden = document.querySelector("button[key='hidden_chatbot']");
                if (hidden) hidden.click();
            };
        </script>
        """,
        unsafe_allow_html=True,
    )

elif st.session_state.page == "chatbot":
    with st.container():
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-title">🤖 AI Heat Risk Advisor</div>
                <p class="hero-subtitle">Ask questions about ward heat risk, vulnerable populations, and cooling strategies.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.container():
        if st.button("⬅ Back to Map", on_click=set_page, args=("map",)):
            st.stop()

        user_question = st.text_input(
            "Ask your question:",
            placeholder="Type a ward risk question and press Enter...",
            label_visibility="visible",
        )

        st.write("**Quick questions:**")
        q1, q2, q3 = st.columns(3)
        quick_questions = {
            "Elderly Risk": "Which areas are most dangerous for elderly people?",
            "Cooling Needed": "Which wards need urgent cooling measures?",
            "Tree Planting": "Where should trees be planted first?",
        }
        if q1.button("👴 Elderly Risk"):
            user_question = quick_questions["Elderly Risk"]
        if q2.button("❄️ Cooling Needed"):
            user_question = quick_questions["Cooling Needed"]
        if q3.button("🌳 Tree Planting"):
            user_question = quick_questions["Tree Planting"]

        if user_question and user_question != st.session_state.last_question:
            with st.spinner("Analyzing heat risk..."):
                context = build_context(processed_df)
                answer = ask_ai(user_question, context, processed_df)
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("AI", answer))
            st.session_state.last_question = user_question

        with st.container():
            st.markdown('<div class="chat-focus">', unsafe_allow_html=True)
            if st.session_state.chat_history:
                for role, text in st.session_state.chat_history:
                    if role == "You":
                        st.markdown(f"**🧑 {role}:** {text}")
                    else:
                        st.markdown(f"**🤖 {role}:** {text}")
            else:
                st.info("💡 Start the conversation by asking a question or using a quick prompt.")

            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.session_state.last_question = ""
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        """
        <div class="chat-focus">
            <strong>Tip:</strong> Use natural language questions like "Which wards are hottest today?" or "What cooling strategies are best for dense neighborhoods?"
        </div>
        """,
        unsafe_allow_html=True,
    )
