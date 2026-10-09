import io

import requests
import streamlit as st
from PIL import Image


API_URL = "http://127.0.0.1:5000"
SESSION_ID = "admin"

FAULT_LABELS = {
    "Outer_race": "Outer race",
    "Inner_race": "Inner race",
    "Bearing_Balls": "Bearing balls",
    "Cage": "Cage",
}

PLOT_LABELS = {
    1: "Outer race envelope spectrum",
    2: "Inner race envelope spectrum",
    3: "Bearing balls envelope spectrum",
    4: "Cage envelope spectrum",
    5: "Frequency-domain XAI matrix",
    6: "Time-domain XAI matrix",
    7: "Time feature correlation report",
    8: "Frequency feature correlation report",
}


st.set_page_config(page_title="BEARING-FDD", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #1f2937;
        --muted: #64748b;
        --line: #d8dee8;
        --panel: #f8fafc;
        --blue: #1d4ed8;
        --green: #0f766e;
        --red: #b91c1c;
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .be-header {
        border-bottom: 1px solid var(--line);
        padding: 0.25rem 0 0.9rem 0;
        margin-bottom: 1rem;
    }
    .be-title {
        font-size: 2.1rem;
        line-height: 1.15;
        color: var(--ink);
        font-weight: 760;
    }
    .be-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-top: 0.25rem;
    }
    .status-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.5rem 0 1rem 0;
    }
    .status-box {
        border: 1px solid var(--line);
        background: white;
        border-radius: 6px;
        padding: 0.8rem 0.9rem;
        min-height: 82px;
    }
    .status-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 700;
    }
    .status-value {
        color: var(--ink);
        font-size: 1.05rem;
        margin-top: 0.25rem;
        font-weight: 720;
    }
    .fault-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.4rem 0 1rem 0;
    }
    .fault-box {
        border: 1px solid var(--line);
        border-left: 4px solid var(--muted);
        background: white;
        border-radius: 6px;
        padding: 0.75rem 0.85rem;
    }
    .fault-on {
        border-left-color: var(--red);
        background: #fff7f7;
    }
    .fault-off {
        border-left-color: var(--green);
        background: #f7fffd;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.84rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.7rem 0.8rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=10)
def get_json(path):
    response = requests.get(f"{API_URL}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


def post_json(path, payload):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def load_plot(flag):
    response = requests.get(f"{API_URL}/getImage/{SESSION_ID}/{flag}", timeout=20)
    if response.status_code != 200 or not response.content:
        return None
    try:
        return Image.open(io.BytesIO(response.content))
    except Exception:
        return None


def stage_text(result):
    raw = result.get("fault_info") or ""
    if "early" in raw:
        return "Early degradation"
    if "medium" in raw:
        return "Medium degradation"
    if "last" in raw:
        return "Last degradation"
    return "Healthy / not confirmed"


def render_fault_boxes(fault_types):
    st.markdown('<div class="fault-grid">', unsafe_allow_html=True)
    for key, label in FAULT_LABELS.items():
        active = key in fault_types
        klass = "fault-box fault-on" if active else "fault-box fault-off"
        value = "Detected" if active else "Not detected"
        st.markdown(
            f"""
            <div class="{klass}">
                <div class="status-label">{label}</div>
                <div class="status-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_plot(flag, width=True):
    img = load_plot(flag)
    if img is None:
        st.info(f"{PLOT_LABELS.get(flag, f'plot{flag}')} is not available for this run.")
        return
    st.image(img, caption=PLOT_LABELS.get(flag, f"plot{flag}"), use_container_width=width)


try:
    datasets = get_json("/getModelsList")["modelsList"]
except Exception as exc:
    st.error(f"Backend is not available at {API_URL}. Start Flask first. Details: {exc}")
    st.stop()


st.markdown(
    """
    <div class="be-header">
      <div class="be-title">BEARING-FDD</div>
      <div class="be-subtitle">Early bearing fault detection and diagnosis using project MS2AE models</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Analysis setup")
    dataset = st.selectbox("Preloaded dataset", datasets, index=datasets.index("IMS2") if "IMS2" in datasets else 0)
    info = get_json(f"/getDatasetByName/{dataset}")

    st.divider()
    st.subheader("Operating condition")
    shaft = st.number_input("Shaft frequency (Hz)", value=float(info["shaft_frequency"]))
    fs = st.number_input("Sampling frequency (Hz)", value=float(info["sampling_frequency"]))
    load = st.text_input("Load / condition", value=str(info.get("carga", "")))

    st.subheader("Bearing geometry")
    bpfo = st.number_input("BPFO (Hz)", value=float(info["bpfo"]))
    bpfi = st.number_input("BPFI (Hz)", value=float(info["bpfi"]))
    bsf = st.number_input("BSF (Hz)", value=float(info["bsf"]))
    ftf = st.number_input("FTF (Hz)", value=float(info["ftf"]))

    st.subheader("Samples")
    max_sample = int(info["max_to_check"])
    healthy_n = st.number_input("Healthy samples", min_value=30, max_value=max_sample, value=min(300, max_sample), step=10)
    defaults = {
        "IMS1": 1850,
        "IMS2": 530,
        "IMS3": 5960,
        "XJTU2-1": 450,
        "XJTU2-3": 301,
        "XJTU3-1": 2340,
        "XJTU3-4": 1410,
    }
    first_default = min(defaults.get(dataset, int(healthy_n)), max_sample - 5)
    first_sample = st.number_input("First analyzed sample", min_value=0, max_value=max_sample - 5, value=first_default, step=1)
    analyzed_n = st.number_input("Number of analyzed samples", min_value=5, max_value=max(5, max_sample - int(first_sample)), value=20, step=1)

    run = st.button("Run diagnosis", type="primary", width="stretch")


top_cols = st.columns([0.22, 0.22, 0.22, 0.34])
top_cols[0].metric("Dataset", dataset)
top_cols[1].metric("Available samples", f"{int(info['max_to_check']):,}")
top_cols[2].metric("Healthy samples", int(healthy_n))
top_cols[3].code(f"{dataset}.windowed_ms2ae_autoencoder.keras\nHI mode: windowed_recon_p95", language="text")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
    st.session_state.last_payload = None

if run:
    payload = {
        "nombre_req": dataset,
        "shaft_frequency_req": shaft,
        "sampling_frequency_req": fs,
        "bpfo_req": bpfo,
        "bpfi_req": bpfi,
        "bsf_req": bsf,
        "ftf_req": ftf,
        "healthy_number_req": int(healthy_n),
        "analyzed_number_req": int(analyzed_n),
        "first_sample_req": int(first_sample),
    }
    with st.spinner("Running MS2AE, thresholding, kurtogram, envelope FFT and XAI..."):
        try:
            st.session_state.last_result = post_json(f"/analyzeData/{SESSION_ID}/0", payload)
            st.session_state.last_payload = payload
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

result = st.session_state.last_result
payload = st.session_state.last_payload

if result is None:
    st.info("Select a dataset and run diagnosis. Suggested first check: IMS2, sample #530, 20 analyzed samples.")
    st.stop()

fault_types = result.get("fault_type", [])
detected = bool(result.get("fault_detected"))

status_cols = st.columns(4)
status_cols[0].metric("Fault status", "Detected" if detected else "Not detected")
status_cols[1].metric("Health stage", stage_text(result))
status_cols[2].metric("Fault type count", len(fault_types))
status_cols[3].metric(
    "Analyzed range",
    f"#{payload['first_sample_req']} - #{payload['first_sample_req'] + payload['analyzed_number_req']}",
)

render_fault_boxes(fault_types)

tabs = st.tabs(["Diagnosis", "Spectra", "Explainability", "Raw response"])

with tabs[0]:
    left, right = st.columns([0.44, 0.56], gap="large")
    with left:
        st.subheader("Diagnosis summary")
        if detected:
            st.success(result.get("analysis_result", "A fault has been detected"))
        else:
            st.info("No fault was confirmed in the selected window.")
        st.write("Detected fault families:")
        if fault_types:
            for ft in fault_types:
                st.markdown(f"- **{FAULT_LABELS.get(ft, ft)}**")
        else:
            st.markdown("- None")

        st.write("Stage thresholds:")
        st.json(result.get("stage_thresholds", {}), expanded=False)
    with right:
        st.subheader("Characteristic harmonics")
        details = result.get("fault_details", [])
        if details:
            for ft, values in zip(fault_types, details):
                label = FAULT_LABELS.get(ft, ft)
                values_txt = ", ".join(f"{float(v):.2f}" for v in values)
                st.markdown(f"**{label}:** {values_txt} Hz")
        else:
            st.write("No characteristic harmonics were returned.")

with tabs[1]:
    st.subheader("Envelope FFT by fault family")
    cols = st.columns(2)
    with cols[0]:
        render_plot(1)
        render_plot(3)
    with cols[1]:
        render_plot(2)
        render_plot(4)

with tabs[2]:
    st.subheader("XAI based on engineering features and correlation")
    st.markdown(
        '<div class="small-note">The project explains HI through correlations with time-domain and frequency-domain features, not SHAP/LIME.</div>',
        unsafe_allow_html=True,
    )
    corr_cols = st.columns(2)
    with corr_cols[0]:
        render_plot(6)
        render_plot(7)
    with corr_cols[1]:
        render_plot(5)
        render_plot(8)

    report_cols = st.columns(2)
    report_cols[0].write("Time-domain report")
    report_cols[0].json(result.get("resultTimeReport", []), expanded=False)
    report_cols[1].write("Frequency-domain report")
    report_cols[1].json(result.get("resultFreqReport", []), expanded=False)

with tabs[3]:
    st.json(result, expanded=True)

