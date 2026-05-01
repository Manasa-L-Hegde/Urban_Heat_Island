# 🌡️ Urban Heat Island — Hyperlocal Mapping

Data-driven Streamlit application that identifies and visualizes urban heat islands at the ward level. The app fuses satellite thermal imagery, ground weather station readings, and ward-level attributes to produce interactive heat-risk maps, ward dashboards, and actionable green-cover recommendations.

------------------------------------

## Quick overview
- Combines satellite + weather + built-environment data into a ward-level `Final_Temperature`.
- Computes a `Heat_Index` and `Vulnerability_Index`, assigns `Vulnerability_Category` (Low/Medium/High).
- Streamlit dashboard with interactive Folium map, ward chooser, charts, and an AI heat-risk advisor.

## Features
- Ward selector with per-ward metrics: heat index, temperature, vulnerability, green cover, and recommendations
- Interactive Folium map centered on selected ward
- Altair charts updated for the selected ward and filtered wards
- Green cover recommendation generator and simple ML risk model
- LLM-based advisor for natural-language heat-risk questions (configurable APIs)

## Quickstart (Windows)
1. Clone the repo and create a virtual environment

```bash
git clone https://github.com/Manasa-L-Hegde/Urban_Heat_Island.git
cd Urban_Heat_Island
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Generate or place dataset

```bash
python -m src.data_generation
```

3. Run the dashboard

```bash
streamlit run app/app.py
```

## Project layout
- `app/` — Streamlit UI (`app.py`) and helpers
- `src/` — data generation, fusion, heat-index, advisor, and map builder
- `data/` — generated or provided ward-level CSV (`ward_data.csv`)
- `models/` — simple ML model for risk labeling
- `visualization/`, `processing/`, `outputs/` — supporting scripts and outputs

## Deployment notes
- The app uses third-party APIs (OpenAI / Google Generative AI) optionally — set credentials as environment variables before deploying.
- For a quick deploy, push to Streamlit Community Cloud, Railway, or Render. Replace the `Live demo` URL above once deployed.

## Contributing
- Fork, create a feature branch, and submit a PR. Please include small, focused changes with tests when possible.

##Demo Link
https://manasa-l-hegde-urban-heat-island-appapp-ijwlht.streamlit.app/

## License
MIT — see the `LICENSE` file.

