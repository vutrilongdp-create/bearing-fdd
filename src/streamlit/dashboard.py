import streamlit as st
import os, sys, warnings
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import argrelextrema

# Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Add src/backend to sys.path so the local backend modules (enter_utils, utils_explainability) can be imported
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
if backend_path not in sys.path:
    sys.path.append(backend_path)

import enter_utils

st.set_page_config(page_title="Bearing Fault Diagnosis System", page_icon="🔬", layout="wide")

# Custom CSS for Scientific Design (Premium Light Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Outfit', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #1a365d;
        font-weight: 700;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 30px;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,242,254,0.3);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA PRESETS -----------------
DATASET_CASES = {
    "IMS1": [
        {"label": "Early stage (#1850 - #1880)", "dataset":"IMS1","healthy_number":300,"first_sample":1850,"analyzed_number":30,"sampling_freq":20480,"shaft_freq":33.33,"BPFO":236,"BPFI":297,"BSF":278,"FTF":15},
        {"label": "Medium stage (#2135 - #2145)", "dataset":"IMS1","healthy_number":300,"first_sample":2135,"analyzed_number":10,"sampling_freq":20480,"shaft_freq":33.33,"BPFO":236,"BPFI":297,"BSF":278,"FTF":15},
        {"label": "Last stage (#2150 - #2156)", "dataset":"IMS1","healthy_number":300,"first_sample":2150,"analyzed_number":6,"sampling_freq":20480,"shaft_freq":33.33,"BPFO":236,"BPFI":297,"BSF":278,"FTF":15}
    ],
    "IMS2": [
        {"label": "Early stage (#530 - #550)", "dataset":"IMS2","healthy_number":300,"first_sample":530,"analyzed_number":20,"sampling_freq":20480,"shaft_freq":33.33,"BPFO":236,"BPFI":297,"BSF":278,"FTF":15},
        {"label": "Medium stage (#865 - #875)", "dataset":"IMS2","healthy_number":300,"first_sample":865,"analyzed_number":10,"sampling_freq":20480,"shaft_freq":33.33,"BPFO":236,"BPFI":297,"BSF":278,"FTF":15},
        {"label": "Last stage (#970 - #980)", "dataset":"IMS2","healthy_number":300,"first_sample":970,"analyzed_number":10,"sampling_freq":20480,"shaft_freq":33.33,"BPFO":236,"BPFI":297,"BSF":278,"FTF":15}
    ],
    "IMS3": [
        {"label": "Early stage (#5960 - #5990)", "dataset":"IMS3","healthy_number":300,"first_sample":5960,"analyzed_number":30,"sampling_freq":20480,"shaft_freq":33.33,"BPFO":236,"BPFI":297,"BSF":278,"FTF":15},
        {"label": "Medium stage (#6170 - #6190)", "dataset":"IMS3","healthy_number":300,"first_sample":6170,"analyzed_number":20,"sampling_freq":20480,"shaft_freq":33.33,"BPFO":236,"BPFI":297,"BSF":278,"FTF":15},
        {"label": "Last stage (#6314 - #6324)", "dataset":"IMS3","healthy_number":300,"first_sample":6314,"analyzed_number":10,"sampling_freq":20480,"shaft_freq":33.33,"BPFO":236,"BPFI":297,"BSF":278,"FTF":15}
    ],
    "XJTU2-1": [
        {"label": "Medium stage (#450 - #457)", "dataset":"XJTU2-1","healthy_number":300,"first_sample":450,"analyzed_number":7,"sampling_freq":25600,"shaft_freq":37.5,"BPFO":112.19,"BPFI":178.94,"BSF":75.21,"FTF":14.20},
        {"label": "Medium stage (#455 - #460)", "dataset":"XJTU2-1","healthy_number":300,"first_sample":455,"analyzed_number":5,"sampling_freq":25600,"shaft_freq":37.5,"BPFO":112.19,"BPFI":178.94,"BSF":75.21,"FTF":14.20},
        {"label": "Last stage (#460 - #470)", "dataset":"XJTU2-1","healthy_number":300,"first_sample":460,"analyzed_number":10,"sampling_freq":25600,"shaft_freq":37.5,"BPFO":112.19,"BPFI":178.94,"BSF":75.21,"FTF":14.20}
    ],
    "XJTU2-3": [
        {"label": "Early stage (#301 - #311)", "dataset":"XJTU2-3","healthy_number":300,"first_sample":301,"analyzed_number":10,"sampling_freq":25600,"shaft_freq":37.5,"BPFO":112.19,"BPFI":178.94,"BSF":75.21,"FTF":14.20},
        {"label": "Medium stage (#410 - #420)", "dataset":"XJTU2-3","healthy_number":300,"first_sample":410,"analyzed_number":10,"sampling_freq":25600,"shaft_freq":37.5,"BPFO":112.19,"BPFI":178.94,"BSF":75.21,"FTF":14.20},
        {"label": "Medium stage (#525 - #532)", "dataset":"XJTU2-3","healthy_number":300,"first_sample":525,"analyzed_number":7,"sampling_freq":25600,"shaft_freq":37.5,"BPFO":112.19,"BPFI":178.94,"BSF":75.21,"FTF":14.20}
    ],
    "XJTU3-1": [
        {"label": "Early stage (#2340 - #2360)", "dataset":"XJTU3-1","healthy_number":300,"first_sample":2340,"analyzed_number":20,"sampling_freq":25600,"shaft_freq":40,"BPFO":123.2,"BPFI":196.49,"BSF":82.58,"FTF":15.40},
        {"label": "Medium stage (#2440 - #2460)", "dataset":"XJTU3-1","healthy_number":300,"first_sample":2440,"analyzed_number":20,"sampling_freq":25600,"shaft_freq":40,"BPFO":123.2,"BPFI":196.49,"BSF":82.58,"FTF":15.40},
        {"label": "Last stage (#2520 - #2540)", "dataset":"XJTU3-1","healthy_number":300,"first_sample":2520,"analyzed_number":20,"sampling_freq":25600,"shaft_freq":40,"BPFO":123.2,"BPFI":196.49,"BSF":82.58,"FTF":15.40}
    ],
    "XJTU3-4": [
        {"label": "Early stage (#1410 - #1425)", "dataset":"XJTU3-4","healthy_number":300,"first_sample":1410,"analyzed_number":15,"sampling_freq":25600,"shaft_freq":40,"BPFO":123.2,"BPFI":196.49,"BSF":82.58,"FTF":15.40},
        {"label": "Medium stage (#1445 - #1460)", "dataset":"XJTU3-4","healthy_number":300,"first_sample":1445,"analyzed_number":15,"sampling_freq":25600,"shaft_freq":40,"BPFO":123.2,"BPFI":196.49,"BSF":82.58,"FTF":15.40},
        {"label": "Last stage (#1495 - #1510)", "dataset":"XJTU3-4","healthy_number":300,"first_sample":1495,"analyzed_number":15,"sampling_freq":25600,"shaft_freq":40,"BPFO":123.2,"BPFI":196.49,"BSF":82.58,"FTF":15.40}
    ]
}

st.title("Intelligent Bearing Fault Detection and Diagnosis System")
st.markdown("A Scientific Dashboard utilizing MS2AE for Anomaly Detection and Kurtogram-filtered Envelope Spectrum for Fault Classification.")

with st.sidebar:
    st.header("Configuration")
    
    selected_dataset = st.selectbox("1. Select Dataset", list(DATASET_CASES.keys()))
    
    cases = DATASET_CASES[selected_dataset]
    case_labels = [c["label"] for c in cases]
    selected_label = st.selectbox("2. Select Operating Condition", case_labels)
    
    tc = next(c for c in cases if c["label"] == selected_label)
    
    st.markdown("---")
    st.subheader("System Parameters")
    st.write(f"- **Sampling Frequency:** {tc['sampling_freq']} Hz")
    st.write(f"- **Shaft Frequency:** {tc['shaft_freq']} Hz")
    
    st.subheader("Kinematic Fault Frequencies")
    st.write(f"- **BPFO (Outer):** {tc['BPFO']} Hz")
    st.write(f"- **BPFI (Inner):** {tc['BPFI']} Hz")
    st.write(f"- **BSF (Roller):** {tc['BSF']} Hz")
    st.write(f"- **FTF (Cage):** {tc['FTF']} Hz")
    
    st.markdown("---")
    st.subheader("File Range (Khoảng File)")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        custom_first_sample = st.number_input("First Sample", min_value=0, max_value=100000, value=tc["first_sample"])
    with col_f2:
        default_last_sample = tc["first_sample"] + tc["analyzed_number"]
        if default_last_sample <= custom_first_sample:
            default_last_sample = custom_first_sample + 1
        custom_last_sample = st.number_input("Last Sample", min_value=custom_first_sample + 1, max_value=100000, value=default_last_sample)
        
    custom_analyzed_number = custom_last_sample - custom_first_sample
        
    st.markdown("---")
    st.subheader("Analysis Mode")
    auto_detect = st.checkbox("Auto-detect First Faulty Sample", value=True, help="If checked, automatically diagnoses the first sample that exceeds the safe threshold.")
    if not auto_detect:
        manual_sample = st.number_input("Manual Sample Index", min_value=0, max_value=custom_analyzed_number-1, value=0)
    else:
        manual_sample = None

    st.markdown("---")
    run_btn = st.button("Execute Diagnostic Pipeline", use_container_width=True, type="primary")

@st.cache_resource
def load_keras_model(model_path):
    custom_objects = {'MonotonicityLayer2': enter_utils.MonotonicityLayer2,
                      'SmoothingLayer': enter_utils.SmoothingLayer,
                      'from_config': enter_utils.from_config}
    return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

if run_btn:
    with st.spinner("Analyzing high-frequency vibration signals..."):
        original_cwd = os.getcwd()
        try:
            cwd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
            os.chdir(cwd_path)
            
            # Load Data
            healthy_samples, denoised_healthy = enter_utils.getDataset(tc["dataset"], tc["healthy_number"], 0)
            analyzed_samples = enter_utils.getDataset(tc["dataset"], custom_analyzed_number, custom_first_sample)[0]
            
            if analyzed_samples.size == 0:
                st.error(f"Error: Could not load data for dataset {tc['dataset']} at sample {custom_first_sample}. Please check if the file exists or the range is correct.")
                st.stop()
                
            # Load Model
            model_name = f"{tc['dataset']}.h5"
            model_path = os.path.join('prog_analizador', 'models', model_name)
            
            if not os.path.exists(model_path):
                st.error(f"Model {model_name} not found! Please pre-train it first.")
                st.stop()
                
            ms2ae_model = load_keras_model(model_path)
            
            # Extract encoder to get 1D Health Index
            try:
                encoder = keras.models.Model(inputs=ms2ae_model.input, outputs=ms2ae_model.get_layer('latent_hi').output)
            except ValueError:
                encoder = ms2ae_model
            
            HI_healthy = encoder.predict(healthy_samples, verbose=0, batch_size=64)
            HI_analyzed = encoder.predict(analyzed_samples, verbose=0, batch_size=32)
            threshold = enter_utils.getThreshold(HI_healthy)
            
            isFaulty, faultySample = enter_utils.checkStage(HI_analyzed, threshold)
            
            target_sample = faultySample if auto_detect else manual_sample
            
            st.markdown("---")
            st.header("📊 Diagnostic Report")
            
            tab1, tab2, tab3 = st.tabs(["📋 Dashboard", "📈 Spectral Analysis", "🧠 Explainable AI"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Anomaly Detection (MS2AE)")
                    if auto_detect:
                        if isFaulty:
                            st.error(f"⚠️ FAULT DETECTED at sample index {faultySample} in the analyzed range!")
                        else:
                            st.success("✅ SYSTEM HEALTHY. No threshold violation detected.")
                    else:
                        status_text = "⚠️ EXCEEDS THRESHOLD" if HI_analyzed[target_sample].item() >= threshold else "✅ WITHIN SAFE LIMITS"
                        st.info(f"Manual Inspection of sample index {target_sample}. Status: {status_text}")
                        
                with col2:
                    st.subheader("Health Index Status")
                    st.markdown(f'''
                        <div class="metric-card">
                            <h4 style="color:#4a5568; margin-bottom:0;">Safe Threshold</h4>
                            <h2 style="color:#e53e3e; margin-top:0;">{threshold:.4f}</h2>
                        </div>
                    ''', unsafe_allow_html=True)
                    if not auto_detect:
                        st.markdown(f'''
                            <div class="metric-card" style="margin-top:15px;">
                                <h4 style="color:#4a5568; margin-bottom:0;">HI at Sample {target_sample}</h4>
                                <h2 style="color:#3182ce; margin-top:0;">{HI_analyzed[target_sample].item():.4f}</h2>
                            </div>
                        ''', unsafe_allow_html=True)
            
            if target_sample is not None and (not auto_detect or isFaulty):
                # Kurtogram & FFT
                fs = float(tc["sampling_freq"])
                diff_harmonics = enter_utils.differenceSignals(denoised_healthy.flatten(), analyzed_samples[target_sample])
                
                with st.spinner("Computing Spectral Kurtosis (Python)..."):
                    kurtogram = enter_utils.computeKurtogram(diff_harmonics, fs, 3.0)
                    fstart, fend = enter_utils.getFilterBands(kurtogram, fs, 3)
                
                with tab2:
                    st.info(f"Optimal Bandpass Filter applied: **{fstart:.2f} Hz - {fend:.2f} Hz**")
                    
                    fft_f, freqs = enter_utils.filteredFFT(4, fs, fstart, fend, diff_harmonics)
                    
                    # Diagnosis
                    freq_interest = [float(tc["BPFO"]), float(tc["BPFI"]), float(tc["BSF"]), float(tc["FTF"])]
                    output_dir = os.path.join("test_results_verification", "streamlit_run")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    det = enter_utils.determineFailure(output_dir, diff_harmonics, HI_healthy, HI_analyzed[target_sample], fs, fstart, fend, freq_interest, 5.0)
                    
                    st.subheader("🔍 Fault Classification")
                    if len(det.get("fault_type", [])) == 0 or "Unknown" in det.get("fault_type", []):
                        st.warning("⚠️ **UNKNOWN FAULT**: Degradation detected, but spectral peaks do not match predefined bearing faults (Inner/Outer/Roller/Cage). This is often due to extreme noise attenuation or complex compound faults.")
                    else:
                        st.error(f"🚨 **CONFIRMED FAULTS**: {', '.join(det['fault_type'])}")
                        
                    # PLOT: Envelope Spectrum
                    st.subheader("📈 Envelope Spectrum (Frequency Domain)")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=freqs, y=fft_f, mode='lines', name='Filtered Spectrum', line=dict(color='#2b6cb0')))
                    
                    # Add expected fault lines ONLY for detected faults
                    fault_colors = {"BPFO": "#e53e3e", "BPFI": "#dd6b20", "BSF": "#d69e2e", "FTF": "#38a169"}
                    fault_vals = {"BPFO": tc["BPFO"], "BPFI": tc["BPFI"], "BSF": tc["BSF"], "FTF": tc["FTF"]}
                    
                    detected_faults = det.get('fault_type', [])
                    fault_mapping = {
                        "Outer_race": "BPFO",
                        "Inner_race": "BPFI",
                        "Bearing_Balls": "BSF",
                        "Cage": "FTF"
                    }
                    
                    for fault_label in detected_faults:
                        if fault_label in fault_mapping:
                            fault_name = fault_mapping[fault_label]
                            freq_base = fault_vals[fault_name]
                            for k in range(1, 4): # Plot first 3 harmonics
                                fig.add_vline(x=freq_base * k, line_width=2, line_dash="dash", line_color=fault_colors[fault_name], 
                                              annotation_text=f"{fault_name} {k}x" if k==1 else "", annotation_position="top right")
                            
                    fig.update_layout(
                        xaxis_title="Frequency (Hz)",
                        yaxis_title="Amplitude",
                        template="plotly_white",
                        height=500,
                        xaxis=dict(range=[0, 1000]), # Zoom in to relevant frequencies
                        font=dict(family="Outfit")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab3:
                    st.header("🧠 Explainable AI (XAI) Analysis")
                    st.write("Pearson Correlation Matrix showing the linear relationship between the AI's Health Index (HI) and physical energy features (e.g. RMS) across the analyzed range.")
                    
                    with st.spinner("Computing XAI Correlation Matrix..."):
                        import xai_utils
                        # Compute correlation matrix
                        corr_df = xai_utils.compute_correlation_matrix(analyzed_samples, HI_analyzed.flatten())
                        
                        # Reverse Y-axis categories for standard matrix display
                        y_cols = corr_df.columns.tolist()
                        y_cols.reverse()
                        z_data = corr_df.loc[y_cols].values
                        
                        # Create Plotly Heatmap
                        fig_xai = go.Figure(data=go.Heatmap(
                            z=z_data,
                            x=corr_df.columns,
                            y=y_cols,
                            colorscale='Teal',
                            zmin=0, zmax=1,
                            text=np.round(z_data, 2),
                            texttemplate="%{text}",
                            hoverinfo="x+y+z"
                        ))
                        fig_xai.update_layout(
                            height=500,
                            width=600,
                            xaxis=dict(title="Features"),
                            yaxis=dict(title="Features"),
                            margin=dict(l=50, r=50, t=30, b=50),
                            font=dict(family="Outfit")
                        )
                        
                        col_x1, col_x2 = st.columns([2, 1])
                        with col_x1:
                            st.plotly_chart(fig_xai, use_container_width=True)
                        with col_x2:
                            st.info("**Explainability Proof:**")
                            hi_rms_corr = corr_df.loc['HI', 'RMS'] if 'RMS' in corr_df.index else 0
                            st.write(f"- **HI vs RMS Correlation: {hi_rms_corr:.2f}**")
                            if hi_rms_corr >= 0.9:
                                st.success("Extremely High Correlation (>0.90)! This proves the AI's Health Index correctly models the physical degradation (vibration energy) of the bearing.")
                            elif hi_rms_corr >= 0.7:
                                st.warning("High Correlation. The AI captures physical degradation reasonably well, though some noise is present.")
                            else:
                                st.error("Low Correlation. The AI's Health Index diverges from physical RMS energy in this specific time window.")
                
        except Exception as e:
            import traceback
            st.error(f"An error occurred: {str(e)}\n\n```python\n{traceback.format_exc()}\n```")
        finally:
            os.chdir(original_cwd)
