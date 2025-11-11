from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

import scanner.detection as detector
import scanner.utils as utils
from scanner.detection import DEFAULT_COLOR_PALETTE

st.set_page_config(
    page_title="Music sheet scanner",
    layout="wide",
)

st.title("Color-aware music reading")
st.markdown(
    """
Upload one or more snapshots (PNG/JPG) or full PDFs.  
The detector will locate every musical symbol, let you pick the highlight colour for each category,  
and produce colour-enhanced sheets to support neurodiverse musicians.
"""
)


def init_state():
    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "class_names" not in st.session_state:
        st.session_state.class_names = []
    if "class_colors" not in st.session_state:
        st.session_state.class_colors = {}
    if "last_signature" not in st.session_state:
        st.session_state.last_signature = None


def decode_uploads(files) -> List[Dict]:
    decoded = []
    for file in files:
        images, labels = utils.load_uploaded_file(file)
        for img, label in zip(images, labels):
            decoded.append({"image": img, "label": label})
    return decoded


def run_inference(pages: List[Dict]) -> Tuple[List[Dict], List[str]]:
    processed = []
    classes = set()
    for page in pages:
        with st.spinner(f"Scanning {page['label']}"):
            prediction = detector.detect_everything(page["image"])
        processed.append({"label": page["label"], "prediction": prediction})
        classes.update(obj.category.name for obj in prediction.object_prediction_list)
    return processed, sorted(classes)


def seed_color_map(class_names: List[str], existing: Dict[str, str]) -> Dict[str, str]:
    color_map = {cls: existing.get(cls) for cls in class_names if existing.get(cls)}
    for idx, cls in enumerate(class_names):
        if cls not in color_map:
            color_map[cls] = DEFAULT_COLOR_PALETTE[idx % len(DEFAULT_COLOR_PALETTE)]
    return color_map


def class_signature(files):
    if not files:
        return "sample"
    return tuple((f.name, f.size, f.type) for f in files)


def ensure_predictions(files):
    signature = class_signature(files)
    if st.session_state.last_signature == signature:
        return

    if files:
        decoded = decode_uploads(files)
    else:
        decoded = [{"image": utils.load_test_image(), "label": "Sample score"}]

    pages, class_names = run_inference(decoded)
    st.session_state.pages = pages
    st.session_state.class_names = class_names
    st.session_state.class_colors = seed_color_map(
        class_names, st.session_state.class_colors
    )
    st.session_state.last_signature = signature


def build_color_sidebar():
    st.sidebar.header("2. Choose colours")
    if not st.session_state.class_names:
        st.sidebar.info("Upload a score to see the detected classes.")
        return {"fill_alpha": 0.25, "show_labels": False}

    opacity = st.sidebar.slider(
        "Highlight opacity", min_value=0.05, max_value=0.6, value=0.25, step=0.05
    )
    show_labels = st.sidebar.checkbox(
        "Show category labels on sheet", value=False, help="Adds text boxes per symbol."
    )

    updated_colors = {}
    for cls in st.session_state.class_names:
        default_color = st.session_state.class_colors.get(cls, DEFAULT_COLOR_PALETTE[0])
        updated_colors[cls] = st.sidebar.color_picker(cls, default_color)

    st.session_state.class_colors = updated_colors
    return {"fill_alpha": opacity, "show_labels": show_labels}


def show_results(display_opts):
    if not st.session_state.pages:
        st.info("Upload music sheets to begin, or we will use a sample PDF.")
        return

    st.subheader("3. Colour-enhanced sheets")
    color_map = st.session_state.class_colors or None
    for entry in st.session_state.pages:
        prediction = entry["prediction"]
        image_array = np.asarray(prediction.image)
        colored = detector.visualize_predictions(
            image_array,
            prediction.object_prediction_list,
            class_colors=color_map,
            hide_labels=not display_opts["show_labels"],
            hide_conf=True,
            rect_th=3,
            fill_alpha=display_opts["fill_alpha"],
        )
        st.markdown(f"**{entry['label']}**")
        st.image(colored, use_column_width=True)
        _show_class_counts(prediction.object_prediction_list)
        st.divider()


def _show_class_counts(predictions):
    counter = Counter(obj.category.name for obj in predictions)
    if not counter:
        st.caption("No symbols detected.")
        return
    data = [{"Symbol": cls, "Count": counter[cls]} for cls in counter]
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )


def main():
    init_state()

    st.sidebar.header("1. Upload your sheets")
    uploaded_files = st.sidebar.file_uploader(
        "PNG, JPG, JPEG or PDF", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True
    )

    ensure_predictions(uploaded_files)
    display_opts = build_color_sidebar()
    show_results(display_opts)


if __name__ == "__main__":
    main()
