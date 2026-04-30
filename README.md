# Urban Heat Island Hyperlocal Mapping System

## Problem Statement
Urban Heat Island (UHI) is the phenomenon where built-up parts of a city become hotter than nearby areas because of dense buildings, limited green cover, impervious surfaces, and trapped heat. This creates real health and infrastructure risks, especially for elderly residents and low-income communities. Municipal corporations often lack a ward-level map showing where these hotspots are concentrated and how they change across seasons.

## Solution Overview
This project fuses satellite thermal imagery, ground-level weather station readings, and building density data to create a hyperlocal city heat map at ward level. The goal is to identify wards under the highest heat stress, overlay that risk with population density, and generate practical actions such as tree planting, reflective roofing, and summer heat advisories.

## Features Implemented (Current Progress)
- Dataset creation at ward level (synthetic generator)
- Data fusion: Satellite + Weather temperature -> `Final_Temperature`
- Heat Index calculation with normalized inputs and vulnerability scoring
- Heat advisory labels based on `Heat_Index` thresholds
- Green cover recommendation system at ward level
- Streamlit dashboard with interactive Folium map, ward search linked to map,
  draggable chatbot button, and AI-assisted insights (LLM and rule-based fallbacks)
- Simple ML model to predict AI risk labels (`models/heat_risk_model.py`)
- Unit tests for core helpers (see `tests/`)

## Installation & Run (local)
1. Create and activate a Python environment (recommended: Anaconda or venv).

	```bash
	python -m venv .venv
	source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
	pip install -r requirements.txt
	```

2. Generate or place the ward dataset:

	```bash
	python -m src.data_generation     # creates data/ward_data.csv (synthetic)
	```

3. Run the Streamlit dashboard:

	```bash
	python -m streamlit run app/app.py
	```

4. (Optional) Run tests:

	```bash
	pytest -q
	```

## Planned work
- Add integration tests, packaging, and a deployment workflow (Streamlit share or Docker)
- Improve LLM prompt safety and cost controls
- Improve map rendering performance for larger datasets

## Tech Stack
Python, Satellite API, Folium, Streamlit, Pandas, NumPy

## Project Structure
- `data/ward_data.csv`: ward-level dataset used by the current phase of the project
- `src/data_generation.py`: creates the synthetic ward dataset
- `src/data_fusion.py`: combines satellite and weather temperatures
- `src/heat_index.py`: computes the heat index and related scores
- `app/app.py`: Streamlit application for the interactive dashboard
- `requirements.txt`: Python dependencies

## Demo Link
- GitHub repository: https://github.com/Manasa-L-Hegde/urban-heat-hyperlocal-mapping

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).
