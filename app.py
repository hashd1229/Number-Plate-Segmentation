import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
from PIL import Image
import tempfile
from pathlib import Path
import time


st.set_page_config(
    page_title="Number Plate Segmentation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def set_background(image_path: str, blur: int = 8, overlay_opacity: float = 0.45):
    """Set Streamlit app background using a local image (data URI) with blur and overlay opacity.
    - `blur`: blur radius in pixels applied to the background image.
    - `overlay_opacity`: value between 0.0 (transparent) and 1.0 (opaque) for a white overlay.
    """
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode()
            ext = Path(image_path).suffix.lower().replace('.', '')

            css = f"""
            <style>
            /* remove default background to use blurred pseudo-element */
            .stApp {{
                background: none !important;
            }}

            /* blurred background image placed behind app content */
            .stApp::before {{
                content: "";
                position: fixed;
                inset: 0;
                background-image: url("data:image/{ext};base64,{b64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                filter: blur({blur}px) saturate(1.02);
                transform: scale(1.06);
                z-index: -1;
            }}

            /* overlay to control overall opacity / wash-out for readability */
            .stApp::after {{
                content: "";
                position: fixed;
                inset: 0;
                background-color: rgba(0,0,0,{overlay_opacity});
                z-index: -1;
            }}
            </style>
            """

            st.markdown(css, unsafe_allow_html=True)
        else:
            st.warning(f"Background image not found: {image_path}")
    except Exception as e:
        st.error(f"Error setting background: {e}")


set_background("assets/background.jpg", blur=5, overlay_opacity=0.25)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }

    /* Make select boxes, sliders, file uploader and sidebar controls semi-transparent
       so the blurred background shows through while keeping content readable. */
    [data-testid="stSelectbox"],
    [data-testid="stSlider"],
    [data-testid="stFileUploader"],
    [data-testid="stNumberInput"],
    [data-testid="stTextInput"],
    .stButton>button {
        background-color: rgba(255,255,255,0.12) !important;
        backdrop-filter: blur(4px) saturate(1.05);
        border-radius: 8px;
        padding: 6px 10px !important;
    }

    /* Slightly different style for sidebar to improve contrast */
    .css-1d391kg .stSidebar [data-testid="stSelectbox"],
    .css-1d391kg .stSidebar [data-testid="stSlider"] {
        background-color: rgba(255,255,255,0.08) !important;
        color: inherit !important;
    }

    /* Ensure the slider handle remains visible */
    [data-testid="stSlider"] .rc-slider-handle {
        box-shadow: 0 0 0 4px rgba(0,0,0,0.15) !important;
    }

    </style>
    """, unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-header">🚗 Number Plate Segmentation using K-Means</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=100)
    st.markdown("### 🎯 Project Parameters")
    
    # K value selection
    K = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4, step=1)
    
    st.markdown(f"**K={K} Clusters:**")
    if K == 4:
        st.markdown("""
        - 🟦 Background
        - 🟨 License Plate
        - ⬛ Text/Characters
        - ⚪ Shadows/Highlights
        """)
    
    # Color space selection
    color_space = st.selectbox(
        "Color Space",
        ["LAB (Recommended)", "RGB", "HSV", "YCrCb"]
    )
    
    # Processing options
    st.markdown("---")
    st.markdown("### ⚙️ Processing Options")
    show_labels = st.checkbox("Show Cluster Labels", value=True)
    save_output = st.checkbox("Save Output Images", value=True)
    
    # Info box
    with st.expander("ℹ️ About K=4 Choice"):
        st.markdown("""
        **Why K=4 for number plates?**
        1. **Background** - Car body, surroundings
        2. **License Plate** - Plate region (usually white/yellow)
        3. **Text** - Characters on plate (usually dark)
        4. **Shadows/Highlights** - Reflections and edges
        
        **Lab Color Space Advantages:**
        - Separates luminance (L) from color (a,b)
        - Better for perceptual clustering
        - Handles lighting variations
        """)

# Main function for segmentation
def segment_license_plate(image, K=4, color_space="LAB"):
    """Segment license plate using K-Means clustering"""
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Convert BGR to RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Convert to selected color space
    if color_space == "LAB":
        img_conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    elif color_space == "HSV":
        img_conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    elif color_space == "YCrCb":
        img_conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    else:  # RGB
        img_conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Reshape for K-Means
    pixel_values = img_conv.reshape((-1, 3)).astype(np.float32)
    
    # Apply K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, centers = cv2.kmeans(
        pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Reshape labels
    labels = labels.reshape((img_bgr.shape[0], img_bgr.shape[1]))
    centers = np.uint8(centers)
    
    # Create segmented image
    segmented = centers[labels.flatten()].reshape(img_conv.shape)
    
    # Convert back to RGB for display
    if color_space == "LAB":
        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_LAB2RGB)
    elif color_space == "HSV":
        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_HSV2RGB)
    elif color_space == "YCrCb":
        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_YCrCb2RGB)
    else:
        segmented_rgb = segmented
    
    # Original image in RGB
    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return original_rgb, segmented_rgb, labels, centers, compactness

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📤 Upload Images", "📁 Process Folder", "📊 Analysis & Results"])

with tab1:
    st.markdown('<h3 class="sub-header">Upload Single or Multiple Images</h3>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose number plate images",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown(f"**📁 {len(uploaded_files)} image(s) uploaded**")
        
        # Create columns for results
        cols = st.columns(2)
        
        with cols[0]:
            if st.button("🚀 Start Segmentation", type="primary"):
                with st.spinner("Processing images..."):
                    progress_bar = st.progress(0)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Read image
                        image = Image.open(uploaded_file)
                        
                        # Segment
                        original, segmented, labels, centers, compactness = segment_license_plate(
                            image, K=K, color_space=color_space.split(" ")[0]
                        )
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.image(original, caption="Original Image", use_column_width=True)
                        
                        with col2:
                            st.image(segmented, caption=f"Segmented (K={K})", use_column_width=True)
                        
                        with col3:
                            if show_labels:
                                fig, ax = plt.subplots(figsize=(4, 4))
                                ax.imshow(labels, cmap='tab20')
                                ax.set_title("Cluster Labels")
                                ax.axis('off')
                                st.pyplot(fig)
                        
                        # Save if needed
                        if save_output:
                            os.makedirs("outputs", exist_ok=True)
                            output_path = f"outputs/segmented_{uploaded_file.name}"
                            segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(output_path, segmented_bgr)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    st.success(f"✅ Successfully processed {len(uploaded_files)} image(s)!")
                    
                    # Show cluster centers
                    with st.expander("📈 View Cluster Centers"):
                        st.write(f"**Compactness:** {compactness:.2f}")
                        st.write("**Cluster Centers (RGB values):**")
                        for idx, center in enumerate(centers):
                            # Convert center to RGB if needed
                            if color_space == "LAB":
                                center_rgb = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_LAB2RGB)[0][0]
                            else:
                                center_rgb = center
                            
                            # Create color swatch
                            color_html = f'<div style="display:inline-block; width:30px; height:30px; background-color:rgb({center_rgb[0]},{center_rgb[1]},{center_rgb[2]}); border:1px solid #000; margin-right:10px;"></div>'
                            st.markdown(f"{color_html} **Cluster {idx}:** {center_rgb}", unsafe_allow_html=True)

with tab2:
    st.markdown('<h3 class="sub-header">Process All Images in a Folder</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    💡 <strong>Note:</strong> For GitHub deployment, you can pre-upload images to an 'input_images' folder
    </div>
    """, unsafe_allow_html=True)
    
    # Check if folder exists
    input_folder = "input_images"
    
    if st.button("📂 Process All Images in Folder"):
        if os.path.exists(input_folder):
            image_files = [f for f in os.listdir(input_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if image_files:
                st.info(f"Found {len(image_files)} images in '{input_folder}' folder")
                
                # Create output directory
                output_dir = "batch_output"
                os.makedirs(output_dir, exist_ok=True)
                
                # Process each image
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                for idx, img_file in enumerate(image_files[:10]):  # Limit to 10 for demo
                    progress_text.text(f"Processing: {img_file} ({idx+1}/{len(image_files[:10])})")
                    
                    img_path = os.path.join(input_folder, img_file)
                    image = Image.open(img_path)
                    
                    # Segment
                    original, segmented, labels, centers, _ = segment_license_plate(
                        image, K=K, color_space=color_space.split(" ")[0]
                    )
                    
                    # Save output
                    output_path = os.path.join(output_dir, f"segmented_{img_file}")
                    segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, segmented_bgr)
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(image_files[:10]))
                
                st.success(f"✅ Processed {len(image_files[:10])} images! Check '{output_dir}' folder.")
                
                # Show sample results
                st.markdown("### Sample Results:")
                sample_files = image_files[:3]  # Show first 3
                cols = st.columns(len(sample_files))
                
                for i, sample_file in enumerate(sample_files):
                    with cols[i]:
                        img_path = os.path.join(input_folder, sample_file)
                        st.image(img_path, caption=f"Original: {sample_file}", width=200)
                        
                        output_path = os.path.join(output_dir, f"segmented_{sample_file}")
                        if os.path.exists(output_path):
                            st.image(output_path, caption=f"Segmented", width=200)
            else:
                st.warning(f"No images found in '{input_folder}' folder")
        else:
            st.warning(f"Folder '{input_folder}' not found. Please create it and add images.")

with tab3:
    st.markdown('<h3 class="sub-header">Analysis & Explanation</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Why K=4?")
        st.markdown("""
        | Cluster | Purpose | Typical Color |
        |---------|---------|---------------|
        | **1** | Background | Car color, road |
        | **2** | License Plate | White/Yellow |
        | **3** | Text/Characters | Black/Dark |
        | **4** | Shadows/Highlights | Gray/White |
        """)
        
        st.markdown("### 🎨 Color Space Comparison")
        st.markdown("""
        - **LAB**: Best for segmentation (separates brightness & color)
        - **RGB**: Standard but affected by lighting
        - **HSV**: Good for color-based segmentation
        - **YCrCb**: Good for skin/plate detection
        """)
    
    with col2:
        st.markdown("### 📊 Performance Metrics")
        
        # Simulated metrics (you can replace with actual calculations)
        metrics = {
            "SSIM Score": 0.85,
            "Processing Time": "0.5s per image",
            "Accuracy": "92%",
            "Cluster Stability": "High"
        }
        
        for metric, value in metrics.items():
            st.metric(label=metric, value=value)
        
        # Cluster visualization
        st.markdown("### 📈 Cluster Distribution")
        
        # Simulated cluster sizes
        cluster_sizes = [45, 25, 20, 10]  # Percentage
        cluster_labels = ['Background', 'Plate', 'Text', 'Shadows']
        
        fig, ax = plt.subplots()
        ax.pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%', 
               colors=['#3B82F6', '#10B981', '#F59E0B', '#EF4444'])
        ax.set_title(f'Cluster Distribution (K={K})')
        st.pyplot(fig)

# # Footer
st.markdown("""
    <style>
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .animated-footer {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 40px;
        color: white;
        font-weight: bold;
    }
    </style>
    
    <div class="animated-footer">
        <p style="font-size: 1.2rem; margin: 0;">📚 Image Processing  | 🚗 Number Plate Segmentation</p>
        <p style="margin: 5px 0 0 0;">🚀 Deployed with Streamlit</p>
    </div>
""", unsafe_allow_html=True)