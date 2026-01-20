import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import folium
    from streamlit_folium import folium_static
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

st.set_page_config(page_title="Fuel Pilferage Tracker", page_icon="⛽", layout="wide")

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIM = 384

class FuelPilferageAnalyzer:

    def __init__(self):
        self.vehicle_data = None
        self.fuel_records = None
        self.site_coordinates = None
        self.analysis_results = None

    def load_data(self, vehicle_file, fuel_file, coordinates_file):
        def read_file(file):
            filename = file.name.lower()
            if filename.endswith('.csv'):
                return pd.read_csv(file)
            elif filename.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format")

        self.vehicle_data = read_file(vehicle_file)
        self.fuel_records = read_file(fuel_file)
        self.site_coordinates = read_file(coordinates_file)
        return True, "Files loaded successfully"

    def preprocess_data(self):
        for df in [self.vehicle_data, self.fuel_records, self.site_coordinates]:
            df.columns = df.columns.str.lower().str.replace(" ", "_")

        # Convert datetime columns
        for df in [self.vehicle_data, self.fuel_records]:
            for col in df.columns:
                if any(k in col for k in ['date', 'time']):
                    df[col] = pd.to_datetime(df[col], errors='coerce')

        return True, "Preprocessing done"

    def analyze_pilferage(self, proximity_radius=100):

        # Detect columns
        v_lat = next(c for c in self.vehicle_data.columns if 'lat' in c)
        v_lon = next(c for c in self.vehicle_data.columns if 'lon' in c)
        v_date = next(c for c in self.vehicle_data.columns if pd.api.types.is_datetime64_any_dtype(self.vehicle_data[c]))
        v_id = next((c for c in self.vehicle_data.columns if 'vehicle' in c), None)
        v_speed = next((c for c in self.vehicle_data.columns if any(k in c for k in ['speed', 'velocity', 'kmh'])), None)

        f_date = next(c for c in self.fuel_records.columns if pd.api.types.is_datetime64_any_dtype(self.fuel_records[c]))
        f_site = next(c for c in self.fuel_records.columns if 'site' in c)

        s_lat = next(c for c in self.site_coordinates.columns if 'lat' in c)
        s_lon = next(c for c in self.site_coordinates.columns if 'lon' in c)
        s_id = next(c for c in self.site_coordinates.columns if 'site' in c)

        site_lookup = {
            row[s_id]: (row[s_lat], row[s_lon])
            for _, row in self.site_coordinates.iterrows()
        }

        genuine, fake, unauthorized = [], [], []

        for _, fuel in self.fuel_records.iterrows():
            site = fuel[f_site]
            date = fuel[f_date]

            if site not in site_lookup:
                fake.append({'site_id': site, 'reason': 'Unknown site'})
                continue

            site_lat, site_lon = site_lookup[site]
            same_day = self.vehicle_data[self.vehicle_data[v_date].dt.date == date.date()]

            valid_presence = False
            closest_vehicle = None
            closest_distance = None

            for _, v in same_day.iterrows():
                lat_diff = v[v_lat] - site_lat
                lon_diff = v[v_lon] - site_lon
                dist = np.sqrt(lat_diff**2 + (lon_diff * np.cos(np.radians(site_lat)))**2) * 111000

                speed = v[v_speed] if v_speed else None

                if dist <= proximity_radius and speed == 0:
                    valid_presence = True
                    closest_vehicle = v.get(v_id, 'Unknown')
                    closest_distance = dist
                    break

            if valid_presence:
                genuine.append({
                    'site_id': site,
                    'vehicle_id': closest_vehicle,
                    'distance_m': round(closest_distance, 2),
                    'status': 'Verified'
                })
            else:
                fake.append({
                    'site_id': site,
                    'reason': 'No stopped vehicle within radius'
                })

        # Unauthorized visits
        for _, v in self.vehicle_data.iterrows():
            speed = v[v_speed] if v_speed else None
            if speed != 0:
                continue

            for site, (slat, slon) in site_lookup.items():
                lat_diff = v[v_lat] - slat
                lon_diff = v[v_lon] - slon
                dist = np.sqrt(lat_diff**2 + (lon_diff * np.cos(np.radians(slat)))**2) * 111000

                if dist <= proximity_radius:
                    unauthorized.append({
                        'site_id': site,
                        'vehicle_id': v.get(v_id, 'Unknown'),
                        'reason': 'Stopped vehicle without fuel record'
                    })

        self.analysis_results = {
            'genuine_entries': genuine,
            'fake_entries': fake,
            'unauthorized_visits': unauthorized
        }

        return self.analysis_results
