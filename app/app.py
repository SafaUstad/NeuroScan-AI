
from datetime import datetime

import streamlit as st
from PIL import Image as PILImage

from predictor import predict_image
from report_generator import generate_pdf_report


st.set_page_config(
    page_title="NeuroScan AI | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------- CUSTOM STYLING --------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F8FC;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }

    .hero {
        background: linear-gradient(120deg, #17365D, #2878C8);
        padding: 26px 30px;
        border-radius: 16px;
        margin-bottom: 10px;
    }

    .hero h1 {
        color: white !important;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #EAF3FC !important;
        font-size: 16px;
        margin-bottom: 0;
    }

    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #17365D !important;
    }

    .stApp p {
        color: #475467;
    }

    .section-card {
        background: white;
        padding: 22px;
        border: 1px solid #D6E5F5;
        border-radius: 14px;
        margin-bottom: 18px;
    }

    .result-card {
        background: white;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #D6E5F5;
        border-left: 5px solid #2878C8;
    }

    .result-label {
        color: #667085 !important;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .result-value {
        color: #17365D !important;
        font-size: 27px;
        font-weight: 700;
        text-transform: capitalize;
    }

    /* Light-blue image upload area */
    div[data-testid="stFileUploader"] {
        background: #EAF3FC;
        border: 1px solid #C7DDF5;
        border-radius: 12px;
        padding: 12px;
    }

    div[data-testid="stFileUploader"] section {
        background: #EAF3FC;
        border: none;
    }

    div[data-testid="stFileUploader"] button {
        background: #D6E9FC;
        color: #17365D;
        border: 1px solid #A8CFF5;
        border-radius: 8px;
    }

    div[data-testid="stFileUploader"] button:hover {
        background: #C2DFFB;
        color: #17365D;
        border-color: #2878C8;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 44px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- HEADER --------------------

st.markdown(
    """
    <div class="hero">
        <h1>🧠 NeuroScan AI</h1>
        <p>
            Brain MRI Analysis Dashboard<br>
            Upload an MRI image to view the classification results
            and generate a detailed report.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- SESSION STATE --------------------

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

if "analysis_image" not in st.session_state:
    st.session_state["analysis_image"] = None

if "analysis_filename" not in st.session_state:
    st.session_state["analysis_filename"] = None


# -------------------- MAIN LAYOUT --------------------

left_col, right_col = st.columns(
    [1, 1.15],
    gap="large",
)


# -------------------- LEFT: IMAGE UPLOAD --------------------

with left_col:
    st.markdown("### Upload MRI Image")
    st.write("Choose an image file to begin the analysis.")

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, and PNG.",
        key="mri_uploader",
    )

    if uploaded_file is not None:
        try:
            image = PILImage.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded MRI Image",
                use_container_width=True,
            )

            st.caption(
                f"File: {uploaded_file.name} · "
                f"Dimensions: {image.width} × {image.height}"
            )

            if st.button(
                "Analyze MRI Image",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Analyzing MRI image... Please wait."):
                        result = predict_image(image)

                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_image"] = image.copy()
                    st.session_state["analysis_filename"] = uploaded_file.name

                    st.rerun()

                except Exception as error:
                    st.error("Unable to analyze this image.")
                    st.exception(error)

        except Exception as error:
            st.error(f"Unable to open this image: {error}")


# -------------------- RIGHT: ANALYSIS RESULTS --------------------

with right_col:
    st.markdown("### Analysis Results")

    result = st.session_state["analysis_result"]
    result_image = st.session_state["analysis_image"]

    if result is None:
        st.markdown(
            """
            <div class="section-card">
                <h4>Ready for Analysis</h4>
                <p>
                    Upload an MRI image and select
                    <b>Analyze MRI Image</b> to display the results here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        # Prediction summary
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Class</div>
                <div class="result-value">
                    {predicted_class.replace("_", " ").title()}
                </div>
                <br>
                <div class="result-label">Model Confidence</div>
                <div class="result-value">{confidence * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("### Class-wise Probability Scores")

        # One set of percentage bars, ordered from highest to lowest.
        chart_data = sorted(
            [
                (name.replace("_", " ").title(), float(value) * 100)
                for name, value in probabilities.items()
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        for class_name, percentage in chart_data:
            label_col, percentage_col = st.columns([3, 1])

            with label_col:
                st.markdown(
                    f"**{class_name}**"
                )

            with percentage_col:
                st.markdown(
                    f"<div style='text-align:right; color:#17365D; "
                    f"font-weight:600;'>{percentage:.2f}%</div>",
                    unsafe_allow_html=True,
                )

            # Native Streamlit progress bar: no custom HTML required.
            st.progress(
                min(100, max(0, int(round(percentage)))),
                text=None,
            )

        st.write("")

        # -------------------- PDF REPORT --------------------

        try:
            pdf_bytes = generate_pdf_report(
                image=result_image,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"NeuroScan_Report_{timestamp}.pdf"

            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

        except Exception as error:
            st.error(f"Could not generate the PDF report: {error}")


# -------------------- FOOTER --------------------

st.markdown(
    """
    <div style="
        text-align:center;
        color:#667085;
        padding:10px;
    ">
        <b style="color:#17365D;">NeuroScan AI</b>
        <br>
        Brain MRI Analysis Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
