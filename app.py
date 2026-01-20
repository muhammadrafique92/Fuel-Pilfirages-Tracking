Fimport streamlit as st
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

st.markdown("""
<style>
.main-header {font-size: 2rem; color: #2E86C1; text-align: center; margin-bottom: 1rem;}
.alert-high {background: #ffebee; border-left: 5px solid #f44336; padding: 0.8rem; margin: 0.3rem 0;}
.alert-medium {background: #fff3e0; border-left: 5px solid #ff9800; padding: 0.8rem; margin: 0.3rem 0;}
.alert-low {background: #e8f5e8; border-left: 5px solid #4caf50; padding: 0.8rem; margin: 0.3rem 0;}
.chat-box {background: #f8f9fa; border: 1px solid #dee2e6; padding: 0.8rem; border-radius: 5px; margin: 0.5rem 0;}
.mode-selector {background: #e3f2fd; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

class FuelPilferageAnalyzer:
    """Core analysis engine - works without LLM"""
    
    def __init__(self):
        self.vehicle_data = None
        self.fuel_records = None
        self.site_coordinates = None
        self.analysis_results = None
        
    def load_data(self, vehicle_file, fuel_file, coordinates_file):
        """Load Excel/CSV files"""
        try:
            def read_file(file):
                filename = file.name.lower()
                if filename.endswith('.csv'):
                    return pd.read_csv(file)
                elif filename.endswith(('.xlsx', '.xls')):
                    return pd.read_excel(file, engine='openpyxl' if filename.endswith('.xlsx') else None)
                else:
                    raise ValueError(f"Unsupported file format: {filename}")
            
            self.vehicle_data = read_file(vehicle_file)
            self.fuel_records = read_file(fuel_file)
            self.site_coordinates = read_file(coordinates_file)
            
            return True, f"Data loaded: {len(self.vehicle_data)} vehicles, {len(self.fuel_records)} fuel records, {len(self.site_coordinates)} sites"
        except Exception as e:
            return False, f"Error loading files: {str(e)}"
    
    def preprocess_data(self):
        """Clean and standardize data"""
        try:
            for df in [self.vehicle_data, self.fuel_records, self.site_coordinates]:
                if df is not None:
                    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
            
            date_keywords = ['date', 'time', 'timestamp']
            for df in [self.vehicle_data, self.fuel_records]:
                if df is not None:
                    for col in df.columns:
                        if any(keyword in col for keyword in date_keywords):
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                            except:
                                continue
            return True, "Data preprocessing completed"
        except Exception as e:
            return False, f"Preprocessing error: {str(e)}"
    
    def analyze_pilferage(self, proximity_radius=100):
        """Core analysis logic - genuine vs fake vs unauthorized"""
        if not all([self.vehicle_data is not None, self.fuel_records is not None, self.site_coordinates is not None]):
            return None
        v_speed = next(
            (c for c in self.vehicle_data.columns 
             if 'speed' in c.lower()), 
             None
             )
        if v_speed is None:
            st.error("Vehicle speed column not found. Speed = 0 is mandatory for validation.")
            return None

        try:
            # Auto-detect required columns
            v_lat = next((c for c in self.vehicle_data.columns if 'lat' in c.lower()), None)
            v_lon = next((c for c in self.vehicle_data.columns if 'lon' in c.lower()), None)
            v_date = next((c for c in self.vehicle_data.columns if pd.api.types.is_datetime64_any_dtype(self.vehicle_data[c])), None)
            v_id = next((c for c in self.vehicle_data.columns if 'vehicle' in c.lower() and 'id' in c.lower()), None)
            
            f_date = next((c for c in self.fuel_records.columns if pd.api.types.is_datetime64_any_dtype(self.fuel_records[c])), None)
            f_site = next((c for c in self.fuel_records.columns if 'site' in c.lower()), None)
            f_amount = next((c for c in self.fuel_records.columns if any(kw in c.lower() for kw in ['amount', 'liter', 'quantity'])), None)
            
            s_lat = next((c for c in self.site_coordinates.columns if 'lat' in c.lower()), None)
            s_lon = next((c for c in self.site_coordinates.columns if 'lon' in c.lower()), None)
            s_id = next((c for c in self.site_coordinates.columns if 'site' in c.lower()), None)
            
            if not all([v_lat, v_lon, v_date, f_date, f_site, s_lat, s_lon, s_id]):
                st.error("Missing required columns in data files")
                return None
            
            # Clean data
            vehicles = self.vehicle_data[
                self.vehicle_data[v_lat].notna() & 
                self.vehicle_data[v_lon].notna() &
                self.vehicle_data[v_date].notna() & 
                (self.vehicle_data[v_lat] != 0)
            ].copy()
            
            # Performance optimization: sample large datasets
            max_vehicles = 2000
            if len(vehicles) > max_vehicles:
                vehicles = vehicles.sample(n=max_vehicles, random_state=42)
                st.info(f"Sampled {max_vehicles} vehicles for performance")
            
            fuels = self.fuel_records[
                self.fuel_records[f_date].notna() & 
                self.fuel_records[f_site].notna()
            ].copy()
            
            sites = self.site_coordinates[
                self.site_coordinates[s_lat].notna() & 
                self.site_coordinates[s_id].notna()
            ].copy()
            
            # Create site lookup for fast access
            site_lookup = {row[s_id]: (row[s_lat], row[s_lon]) for _, row in sites.iterrows()}
            
            # Initialize result containers
            genuine_entries = []
            fake_entries = []
            unauthorized_visits = []
            
            st.info(f"Analyzing {len(fuels)} fuel records...")
            progress_bar = st.progress(0)
            
            # ANALYSIS LOGIC: Check each fuel record
            for idx, fuel_record in fuels.iterrows():
                site_id = fuel_record[f_site]
                fuel_date = fuel_record[f_date]
                amount = fuel_record.get(f_amount, 0)
                
                if site_id not in site_lookup:
                    fake_entries.append({
                        'site_id': site_id,
                        'date': fuel_date.strftime('%Y-%m-%d'),
                        'fuel_amount': amount,
                        'reason': 'Site coordinates not found',
                        'risk_level': 'High'
                    })
                    continue
                
                # Get vehicles that were present on the same date
                same_date_vehicles = vehicles[vehicles[v_date].dt.date == fuel_date.date()]
                
                if same_date_vehicles.empty:
                    fake_entries.append({
                        'site_id': site_id,
                        'date': fuel_date.strftime('%Y-%m-%d'),
                        'fuel_amount': amount,
                        'reason': 'No vehicle GPS data on fuel delivery date',
                        'risk_level': 'High'
                    })
                    continue
                
                # Check if any vehicle was within proximity radius
                site_lat, site_lon = site_lookup[site_id]
                vehicle_found_nearby = False
                closest_distance = float('inf')
                closest_vehicle = 'Unknown'
                
                for _, vehicle in same_date_vehicles.iterrows():
                    # Calculate distance using simple approximation (faster than geopy)
                    lat_diff = vehicle[v_lat] - site_lat
                    lon_diff = vehicle[v_lon] - site_lon
                    distance_meters = np.sqrt(lat_diff**2 + (lon_diff * np.cos(np.radians(site_lat)))**2) * 111000

                    vehicle_speed = vehicle.get(v_speed, None)
                    if (
                        distance_meters <= proximity_radius
                        and vehicle_speed is not None
                        and vehicle_speed == 0
                        ):
                        vehicle_found_nearby = True
                        if distance_meters < closest_distance:
                            closest_distance = distance_meters
                            closest_vehicle = vehicle.get(v_id, 'Unknown')
                            break

                if vehicle_found_nearby:
                    # GENUINE ENTRY: Vehicle was present during fuel delivery
                    genuine_entries.append({
                        'site_id': site_id,
                        'date': fuel_date.strftime('%Y-%m-%d'),
                        'fuel_amount': amount,
                        'vehicle_id': closest_vehicle,
                        'distance_meters': closest_distance,
                        'verification_status': 'Verified'
                    })
                else:
                    # FAKE ENTRY: No vehicle within radius during fuel delivery
                    fake_entries.append({
                        'site_id': site_id,
                        'date': fuel_date.strftime('%Y-%m-%d'),
                        'fuel_amount': amount,
                        'reason': f'No stationary vehicle (speed=0) within {proximity_radius}m radius',
                        'risk_level': 'High'
                    })
                
                progress_bar.progress((idx + 1) / len(fuels))
            
            progress_bar.empty()
            
            # UNAUTHORIZED VISITS: Find vehicles at sites without fuel deliveries
            st.info("Checking for unauthorized visits...")
            fuel_delivery_dates = set((f[f_site], f[f_date].date()) for _, f in fuels.iterrows())
            
            # Sample vehicles for performance
            sample_size = min(800, len(vehicles))
            vehicle_sample = vehicles.sample(n=sample_size, random_state=42) if len(vehicles) > sample_size else vehicles
            
            for _, vehicle in vehicle_sample.iterrows():
                v_date_val = vehicle[v_date]
                if pd.isna(v_date_val):
                    continue
                
                v_lat_val = vehicle[v_lat]
                v_lon_val = vehicle[v_lon]
                v_id_val = vehicle.get(v_id, 'Unknown')
                
                # Check proximity to all sites
                for site_id, (site_lat, site_lon) in site_lookup.items():
                    lat_diff = v_lat_val - site_lat
                    lon_diff = v_lon_val - site_lon
                    distance_meters = np.sqrt(lat_diff**2 + (lon_diff * np.cos(np.radians(site_lat)))**2) * 111000
                    
                    vehicle_speed = vehicle.get(v_speed, None)
                    if (
                        distance_meters <= proximity_radius
                        and vehicle_speed is not None
                        and vehicle_speed == 0
                        ):

                        # Vehicle was at this site - check if there was a fuel delivery
                        if (site_id, v_date_val.date()) not in fuel_delivery_dates:
                            unauthorized_visits.append({
                                'site_id': site_id,
                                'date': v_date_val.strftime('%Y-%m-%d %H:%M'),
                                'vehicle_id': v_id_val,
                                'distance_meters': distance_meters,
                                'reason': 'Vehicle at site without fuel delivery record',
                                'risk_level': 'Medium'
                            })
            
            # Compile results
            self.analysis_results = {
                'genuine_entries': genuine_entries,
                'fake_entries': fake_entries,
                'unauthorized_visits': unauthorized_visits,
                'summary_stats': {
                    'total_fuel_records': len(fuels),
                    'genuine_count': len(genuine_entries),
                    'fake_count': len(fake_entries),
                    'unauthorized_count': len(unauthorized_visits),
                    'genuineness_rate': (len(genuine_entries) / len(fuels)) * 100 if len(fuels) > 0 else 0,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'proximity_radius': proximity_radius
                }
            }
            
            return self.analysis_results
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None
    
    def export_results_to_csv(self):
        """Export analysis results to CSV files"""
        if not self.analysis_results:
            return None
        
        csv_files = {}
        
        if self.analysis_results['genuine_entries']:
            genuine_df = pd.DataFrame(self.analysis_results['genuine_entries'])
            csv_files['genuine_entries.csv'] = genuine_df.to_csv(index=False)
        
        if self.analysis_results['fake_entries']:
            fake_df = pd.DataFrame(self.analysis_results['fake_entries'])
            csv_files['fake_entries.csv'] = fake_df.to_csv(index=False)
        
        if self.analysis_results['unauthorized_visits']:
            unauthorized_df = pd.DataFrame(self.analysis_results['unauthorized_visits'])
            csv_files['unauthorized_visits.csv'] = unauthorized_df.to_csv(index=False)
        
        # Summary report
        stats = self.analysis_results['summary_stats']
        summary_data = {
            'Metric': ['Total Fuel Records', 'Genuine Entries', 'Fake Entries', 'Unauthorized Visits', 
                      'Genuineness Rate (%)', 'Analysis Date', 'Proximity Radius (m)'],
            'Value': [stats['total_fuel_records'], stats['genuine_count'], stats['fake_count'],
                     stats['unauthorized_count'], f"{stats['genuineness_rate']:.2f}",
                     stats['analysis_date'], stats['proximity_radius']]
        }
        summary_df = pd.DataFrame(summary_data)
        csv_files['analysis_summary.csv'] = summary_df.to_csv(index=False)
        
        return csv_files

class RAGQuerySystem:
    """RAG system for natural language queries - requires LLM"""
    
    def __init__(self):
        self.client = None
        self.embedder = None
        self.index = None
        self.knowledge_base = []
        
        if RAG_AVAILABLE and GROQ_API_KEY:
            try:
                self.client = Groq(api_key=GROQ_API_KEY)
                self.embedder = SentenceTransformer(EMBED_MODEL)
            except Exception as e:
                st.warning(f"RAG initialization failed: {e}")
    
    def create_knowledge_base(self, analysis_results):
        """Create searchable knowledge base from analysis results"""
        if not self.embedder or not analysis_results:
            return False
        
        try:
            stats = analysis_results['summary_stats']
            
            # Create comprehensive knowledge chunks
            knowledge_chunks = [
                f"Fuel Pilferage Analysis Summary: Analyzed {stats['total_fuel_records']} fuel delivery records. Found {stats['genuine_count']} genuine entries ({stats['genuineness_rate']:.1f}% verification rate), {stats['fake_count']} fake entries, and {stats['unauthorized_count']} unauthorized vehicle visits. Analysis conducted on {stats['analysis_date']} with {stats['proximity_radius']}m proximity radius.",
                
                f"Genuine Fuel Deliveries: {stats['genuine_count']} fuel deliveries were successfully verified with corresponding vehicle GPS data within the specified radius, indicating legitimate fuel operations.",
                
                f"Fraudulent Activities Detected: {stats['fake_count']} suspicious fuel entries were identified where fuel delivery was recorded but no vehicle was detected at the site location during the delivery time, suggesting potential fuel management system fraud.",
                
                f"Unauthorized Vehicle Activities: {stats['unauthorized_count']} instances of vehicles being detected at fuel sites without corresponding fuel delivery records, indicating possible fuel theft or pilferage attempts.",
                
                f"Risk Assessment: Overall genuineness rate of {stats['genuineness_rate']:.1f}% indicates {'HIGH RISK' if stats['genuineness_rate'] < 70 else 'MEDIUM RISK' if stats['genuineness_rate'] < 85 else 'LOW RISK'} operational environment for fuel management."
            ]
            
            # Add specific examples from results
            if analysis_results['fake_entries']:
                fake_sites = [entry['site_id'] for entry in analysis_results['fake_entries'][:5]]
                knowledge_chunks.append(f"High-risk sites with fake entries include: {', '.join(set(fake_sites))}. These locations require immediate investigation.")
            
            if analysis_results['unauthorized_visits']:
                unauth_sites = [visit['site_id'] for visit in analysis_results['unauthorized_visits'][:5]]
                vehicles = [visit['vehicle_id'] for visit in analysis_results['unauthorized_visits'][:5]]
                knowledge_chunks.append(f"Sites with unauthorized visits: {', '.join(set(unauth_sites))}. Suspicious vehicles: {', '.join(set(vehicles))}.")
            
            # Build FAISS vector index
            vectors = self.embedder.encode(knowledge_chunks, convert_to_numpy=True)
            self.index = faiss.IndexFlatL2(VECTOR_DIM)
            self.index.add(vectors)
            self.knowledge_base = knowledge_chunks
            
            return True
            
        except Exception as e:
            st.error(f"Knowledge base creation failed: {e}")
            return False
    
    def query(self, user_question, top_k=3):
        """Process natural language queries using RAG"""
        if not self.client or not self.index:
            return "RAG system not available. Please check GROQ API key configuration."
        
        try:
            # Retrieve relevant knowledge
            query_vector = self.embedder.encode([user_question], convert_to_numpy=True)
            distances, indices = self.index.search(query_vector, top_k)
            
            relevant_context = "\n".join([self.knowledge_base[i] for i in indices[0]])
            
            # Generate response using GROQ
            messages = [
                {
                    "role": "system", 
                    "content": "You are a fuel pilferage analysis expert. Provide specific, actionable insights based on the analysis data. Be concise but comprehensive."
                },
                {
                    "role": "user",
                    "content": f"Analysis Context:\n{relevant_context}\n\nUser Question: {user_question}"
                }
            ]
            
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_completion_tokens=400,
                stream=False
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"Query processing failed: {str(e)}"

def main():
    st.markdown('<h1 class="main-header"> AI based Fuel Pilferage tracker</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FuelPilferageAnalyzer()
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Mode Selection
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    mode = st.radio(
        "Select Application Mode:",
        options=["Dashboard Mode", "AI Assistant Mode"],
        help="Dashboard Mode: Analysis + visualizations + CSV exports | AI Assistant Mode: Natural language queries with LLM"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    analyzer = st.session_state.analyzer
    
    # Configuration sidebar
    with st.sidebar:
        st.header("Configuration")
        
        with st.expander("System Status"):
            st.write("**Core Features:**")
            st.success("Analysis Engine")
            st.success("Data Processing") 
            st.success("CSV Export")
            
            st.write("**Optional Features:**")
            if PLOTLY_AVAILABLE:
                st.success("Interactive Charts (Plotly)")
            else:
                st.warning("Basic Charts Only")
            
            if mode == "AI Assistant Mode":
                if GROQ_API_KEY and RAG_AVAILABLE:
                    st.success("AI Assistant Ready")
                else:
                    st.error("AI Assistant Unavailable")
    
    # File uploads
    st.sidebar.header("Upload Data Files")
    vehicle_file = st.sidebar.file_uploader("Vehicle GPS Tracking", type=['xlsx', 'xls', 'csv'])
    fuel_file = st.sidebar.file_uploader("Fuel Delivery Records", type=['xlsx', 'xls', 'csv'])  
    coordinates_file = st.sidebar.file_uploader("Site Coordinates", type=['xlsx', 'xls', 'csv'])
    
    if vehicle_file and fuel_file and coordinates_file:
        # Load data if not already loaded
        if analyzer.vehicle_data is None:
            load_success, load_msg = analyzer.load_data(vehicle_file, fuel_file, coordinates_file)
            if load_success:
                st.success(load_msg)
                if analyzer.preprocess_data()[0]:
                    # Data summary
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Vehicle Records", len(analyzer.vehicle_data))
                    col2.metric("Fuel Records", len(analyzer.fuel_records))
                    col3.metric("Sites", len(analyzer.site_coordinates))
                else:
                    st.error("Data preprocessing failed")
                    return
            else:
                st.error(load_msg)
                return
        
        # Analysis controls
        if analyzer.vehicle_data is not None:
            st.header("Analysis Configuration")
            proximity_radius = st.slider("Proximity Radius (meters)", 50, 500, 100, 25)
            
            if st.button("Run Pilferage Analysis", type="primary"):
                with st.spinner("Analyzing fuel pilferage patterns..."):
                    results = analyzer.analyze_pilferage(proximity_radius)
                    
                    if results:
                        st.session_state.analysis_results = results
                        st.session_state.analysis_completed = True
                        st.success("Analysis completed successfully!")
                    else:
                        st.error("Analysis failed. Please check your data format.")
                        return
            
            # Display results if analysis is completed
            if st.session_state.analysis_completed and st.session_state.analysis_results:
                results = st.session_state.analysis_results
                stats = results['summary_stats']
                
                # Display key metrics
                st.header("Analysis Results")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.markdown(f"""
                <div class="alert-low">
                    <h3>Genuine Entries</h3>
                    <h2>{stats['genuine_count']}</h2>
                    <p>{stats['genuineness_rate']:.1f}% verified</p>
                </div>
                """, unsafe_allow_html=True)
                
                col2.markdown(f"""
                <div class="alert-high">
                    <h3>Fake Entries</h3>
                    <h2>{stats['fake_count']}</h2>
                    <p>Potential fraud</p>
                </div>
                """, unsafe_allow_html=True)
                
                col3.markdown(f"""
                <div class="alert-medium">
                    <h3>Unauthorized Visits</h3>
                    <h2>{stats['unauthorized_count']}</h2>
                    <p>Possible theft</p>
                </div>
                """, unsafe_allow_html=True)
                
                risk_level = "Low" if stats['genuineness_rate'] > 85 else "Medium" if stats['genuineness_rate'] > 70 else "High"
                risk_color = "alert-low" if risk_level == "Low" else "alert-medium" if risk_level == "Medium" else "alert-high"
                col4.markdown(f"""
                <div class="{risk_color}">
                    <h3>Risk Level</h3>
                    <h2>{risk_level}</h2>
                    <p>Overall assessment</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Mode-specific interface
                if mode == "Dashboard Mode":
                    display_dashboard_mode(results)
                elif mode == "AI Assistant Mode":
                    display_ai_assistant_mode(results)
    
    else:
        st.info("Please upload all three data files to begin analysis")

def display_dashboard_mode(results):
    """Display dashboard interface"""
    st.header("Dashboard & Exports")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Genuine Entries", "Fake Entries", "Unauthorized Visits", "Export Data"])
    
    with tab1:
        if results['genuine_entries']:
            df = pd.DataFrame(results['genuine_entries'])
            st.subheader("Verified Fuel Deliveries")
            st.dataframe(df, use_container_width=True)
            
            if 'fuel_amount' in df.columns:
                if PLOTLY_AVAILABLE:
                    fig = px.bar(df.groupby('site_id')['fuel_amount'].sum().reset_index(),
                               x='site_id', y='fuel_amount', title="Genuine Fuel Deliveries by Site")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    chart_data = df.groupby('site_id')['fuel_amount'].sum()
                    st.bar_chart(chart_data)
        else:
            st.info("No genuine entries found")
    
    with tab2:
        if results['fake_entries']:
            df = pd.DataFrame(results['fake_entries'])
            st.subheader("Suspected Fake Entries")
            st.dataframe(df, use_container_width=True)
            
            if 'fuel_amount' in df.columns:
                if PLOTLY_AVAILABLE:
                    fig = px.bar(df.groupby('site_id')['fuel_amount'].sum().reset_index(),
                               x='site_id', y='fuel_amount', title="Fake Entries by Site")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    chart_data = df.groupby('site_id')['fuel_amount'].sum()
                    st.bar_chart(chart_data)
        else:
            st.success("No fake entries detected")
    
    with tab3:
        if results['unauthorized_visits']:
            df = pd.DataFrame(results['unauthorized_visits'])
            st.subheader("Unauthorized Vehicle Visits")
            st.dataframe(df, use_container_width=True)
        else:
            st.success("No unauthorized visits detected")
    
    with tab4:
        st.subheader("Export Results")
        analyzer = st.session_state.analyzer
        csv_files = analyzer.export_results_to_csv()
        
        if csv_files:
            for filename, csv_data in csv_files.items():
                st.download_button(
                    label=f"Download {filename}",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv"
                )

def display_ai_assistant_mode(results):
    """Display AI assistant interface"""
    st.header("AI Assistant - Natural Language Queries")
    
    if not GROQ_API_KEY or not RAG_AVAILABLE:
        st.error("AI Assistant not available. Please ensure GROQ_API_KEY is set and RAG dependencies are installed.")
        return
    
    # Initialize RAG system
    if st.session_state.rag_system is None:
        with st.spinner("Initializing AI Assistant..."):
            rag_system = RAGQuerySystem()
            if rag_system.create_knowledge_base(results):
                st.session_state.rag_system = rag_system
                st.success("AI Assistant ready!")
            else:
                st.error("AI Assistant initialization failed")
                return
    
    # Query interface
    user_input = st.text_input("Ask about your analysis:", placeholder="e.g., Which sites show the highest fraud risk?")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Submit Query") and user_input.strip():
            with st.spinner("Processing..."):
                response = st.session_state.rag_system.query(user_input.strip())
                st.session_state.chat_history.append({
                    'question': user_input.strip(),
                    'answer': response
                })
                st.rerun()
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"""
            <div class="chat-box">
                <h4>Q: {chat['question']}</h4>
                <p><strong>AI:</strong> {chat['answer']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
