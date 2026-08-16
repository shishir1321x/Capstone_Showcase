import streamlit as st
import requests
import numpy as np
import pandas as pd
import io

st.set_page_config(page_title="MONAI 3D UNet Viewer", layout="wide")

st.title("🧠 Medical Image Segmentation Dashboard")
st.caption("Streamlit Frontend connected to FastAPI + PyTorch Backend")

# ------------------------------------------------------------------
# 1. Sidebar Configuration & Data Loading
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Server Configuration")
API_URL = st.sidebar.text_input("FastAPI Base URL:", value="http://localhost:8000")

if st.sidebar.button("Check Backend Health"):
    try:
        res = requests.get(f"{API_URL}/health", timeout=5)
        if res.status_code == 200:
            health_data = res.json()
            st.sidebar.success(f"Connected! Device: {health_data.get('device')}")
        else:
            st.sidebar.error(f"Server error status: {res.status_code}")
    except Exception as e:
        st.sidebar.error(f"Connection Failed: {e}")

st.sidebar.divider()
st.sidebar.subheader("📤 Input Volume")

input_source = st.sidebar.radio("Input Source:", ["Generate Synthetic 3D Volume", "Upload .npy File"])

# Initialize session state for volume persistent storage across reruns
if "volume" not in st.session_state:
    st.session_state.volume = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if input_source == "Generate Synthetic 3D Volume":
    if st.sidebar.button("🎲 Generate (64x64x64) Volume"):
        # Create a synthetic 3D volume with a spherical structure inside
        z, y, x = np.ogrid[:64, :64, :64]
        center_z, center_y, center_x = 32, 32, 32
        sphere_mask = (z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2 <= 15**2
        
        # Base noise with embedded synthetic organ intensity
        vol = np.random.normal(0.2, 0.05, (64, 64, 64)).astype(np.float32)
        vol[sphere_mask] += 0.6
        vol = np.clip(vol, 0.0, 1.0)
        
        st.session_state.volume = vol
        st.session_state.prediction = None  # Reset prediction on new data
        st.sidebar.success("Generated synthetic volume!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload 3D Volume (.npy)", type=["npy"])
    if uploaded_file is not None:
        st.session_state.volume = np.load(uploaded_file).astype(np.float32)
        st.sidebar.success(f"Loaded volume shape: {st.session_state.volume.shape}")


# ------------------------------------------------------------------
# 2. Main Dashboard & API Call Trigger
# ------------------------------------------------------------------
vol = st.session_state.volume

if vol is not None:
    depth, height, width = vol.shape
    st.info(f"Loaded 3D Volume Dimensions — **Depth (Z):** {depth} | **Height (Y):** {height} | **Width (X):** {width}")
    
    col_btn, col_blank = st.columns([1, 3])
    with col_btn:
        if st.button("🚀 Run FastAPI Inference"):
            with st.spinner("Transmitting 3D volume to PyTorch server..."):
                try:
                    buffer = io.BytesIO()
                    np.save(buffer, vol)
                    buffer.seek(0)
                    
                    files = {"file": ("volume.npy", buffer.getvalue(), "application/octet-stream")}
                    response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                    
                    if response.status_code == 200:
                        st.session_state.prediction = response.json()
                        st.success("✅ Inference Complete!")
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as err:
                    st.error(f"Failed to connect to backend: {err}")

    # Display Inference Summary if Available
    if st.session_state.prediction:
        res = st.session_state.prediction
        m1, m2, m3 = st.columns(3)
        m1.metric("Latency", f"{res['latency_ms']} ms")
        m2.metric("Target Ratio", f"{res['prediction_summary']['target_ratio'] * 100:.2f}%")
        m3.metric("Target Voxels", f"{res['prediction_summary']['target_organ_voxels']:,}")

    st.divider()

    # ------------------------------------------------------------------
    # 3. Interactive Multi-Planar Slice Scrubbing Section
    # ------------------------------------------------------------------
    st.header("🎛️ Interactive 3D Slicing Inspector")
    
    view_plane = st.radio(
        "Select Viewing Plane:", 
        ["Axial (Z-axis / Top-down)", "Coronal (Y-axis / Front-back)", "Sagittal (X-axis / Side view)"],
        horizontal=True
    )
    
    if view_plane.startswith("Axial"):
        max_slices = depth - 1
        slice_idx = st.slider("Scrub Z-Slice (Axial):", min_value=0, max_value=max_slices, value=max_slices // 2)
        img_slice = vol[slice_idx, :, :]
        plane_title = f"Axial Slice Z = {slice_idx} / {max_slices}"
        
    elif view_plane.startswith("Coronal"):
        max_slices = height - 1
        slice_idx = st.slider("Scrub Y-Slice (Coronal):", min_value=0, max_value=max_slices, value=max_slices // 2)
        img_slice = vol[:, slice_idx, :]
        plane_title = f"Coronal Slice Y = {slice_idx} / {max_slices}"
        
    else:  # Sagittal
        max_slices = width - 1
        slice_idx = st.slider("Scrub X-Slice (Sagittal):", min_value=0, max_value=max_slices, value=max_slices // 2)
        img_slice = vol[:, :, slice_idx]
        plane_title = f"Sagittal Slice X = {slice_idx} / {max_slices}"

    # Render Active Slice Visuals
    col_img, col_hist = st.columns([2, 1])
    
    with col_img:
        st.subheader(plane_title)
        # Display grayscale slice normalized to [0, 255]
        slice_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        st.image(slice_normalized, caption=plane_title, use_container_width=True, clamp=True)
        
    with col_hist:
        st.subheader("Slice Intensity Histogram")
        st.caption("Distribution of pixel/voxel intensities on the active slice:")
        hist_values, bin_edges = np.histogram(img_slice.flatten(), bins=30)
        st.bar_chart(pd.DataFrame({"Voxel Count": hist_values}))

else:
    st.info("👈 Please generate or upload a 3D volume using the sidebar to open the interactive viewer.")
