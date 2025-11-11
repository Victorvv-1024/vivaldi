from collections import Counter
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image
from sahi.prediction import ObjectPrediction
from streamlit_drawable_canvas import st_canvas

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
    if "manual_annotations" not in st.session_state:
        st.session_state.manual_annotations = {}
    if "colored_outputs" not in st.session_state:
        st.session_state.colored_outputs = {}


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
    existing = [cls for cls in st.session_state.class_names if cls not in class_names]
    merged_classes = class_names + existing
    st.session_state.class_names = merged_classes
    st.session_state.class_colors = seed_color_map(
        merged_classes, st.session_state.class_colors
    )
    st.session_state.last_signature = signature


def build_color_sidebar():
    st.sidebar.header("2. Choose colours")
    if not st.session_state.class_names:
        st.sidebar.info("Upload a score to see the detected classes.")
        return {"fill_alpha": 0.25, "show_labels": False}

    _render_custom_class_form()

    opacity = st.sidebar.slider(
        "Fill opacity", min_value=0.05, max_value=0.9, value=0.35, step=0.05
    )
    hide_outline = st.sidebar.checkbox(
        "Hide bounding-box outlines", value=False
    )
    show_labels = st.sidebar.checkbox(
        "Show category labels on sheet", value=False, help="Adds text boxes per symbol."
    )

    updated_colors = {}
    for cls in st.session_state.class_names:
        default_color = st.session_state.class_colors.get(cls, DEFAULT_COLOR_PALETTE[0])
        updated_colors[cls] = st.sidebar.color_picker(cls, default_color)

    st.session_state.class_colors = updated_colors
    return {"fill_alpha": opacity, "show_labels": show_labels, "hide_outline": hide_outline}


def show_results(display_opts):
    if not st.session_state.pages:
        st.info("Upload music sheets to begin, or we will use a sample PDF.")
        return

    st.subheader("3. Colour-enhanced sheets")
    color_map = st.session_state.class_colors or None
    st.session_state.colored_outputs = {}
    download_candidates = []
    for entry in st.session_state.pages:
        prediction = entry["prediction"]
        image_array = np.asarray(prediction.image)
        manual_predictions = _manual_predictions(entry["label"])
        combined_predictions = list(prediction.object_prediction_list) + manual_predictions
        colored = detector.visualize_predictions(
            image_array,
            combined_predictions,
            class_colors=color_map,
            hide_labels=not display_opts["show_labels"],
            hide_conf=True,
            rect_th=0 if display_opts["hide_outline"] else 3,
            fill_alpha=display_opts["fill_alpha"],
        )
        st.markdown(f"**{entry['label']}**")
        st.image(colored, use_column_width=True)
        st.session_state.colored_outputs[entry["label"]] = colored
        download_candidates.append((entry["label"], colored))
        _render_download_button(entry["label"], colored)
        _render_annotation_panel(entry, color_map)
        _show_class_counts(prediction.object_prediction_list)
        st.divider()

    if download_candidates:
        _render_pdf_download(download_candidates)


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


def _render_custom_class_form():
    with st.sidebar.expander("Add custom symbol category"):
        new_label = st.text_input("Label", key="new_class_label")
        new_color = st.color_picker("Colour", value="#ff66c4", key="new_class_color")
        submitted = st.button("Add category", key="add_class_btn")
        if submitted:
            label = new_label.strip()
            if not label:
                st.warning("Please enter a label for the category.")
            elif label in st.session_state.class_names:
                st.info("Category already exists.")
            else:
                st.session_state.class_names.append(label)
                st.session_state.class_colors[label] = new_color
                st.success(f"Added category '{label}'.")


def _manual_predictions(page_label: str) -> List[ObjectPrediction]:
    annotations = st.session_state.manual_annotations.get(page_label, [])
    manual_objs = []
    for idx, item in enumerate(annotations):
        manual_objs.append(
            ObjectPrediction(
                bbox=item["bbox"],
                category_id=-1,
                category_name=item["category"],
                score=1.0,
            )
        )
    return manual_objs


def _render_annotation_panel(entry, color_map):
    label = entry["label"]
    image_array = np.asarray(entry["prediction"].image)
    if label not in st.session_state.manual_annotations:
        st.session_state.manual_annotations[label] = []

    with st.expander(f"Add or edit symbols for {label}"):
        available_classes = st.session_state.class_names or ["custom"]
        selected_class = st.selectbox(
            "Category", available_classes, key=f"class_select_{label}"
        )
        session_colors = st.session_state.class_colors
        default_color = session_colors.get(
            selected_class,
            (color_map or {}).get(selected_class, "#ffaa00"),
        )
        selected_color = st.color_picker(
            "Annotation colour", value=default_color, key=f"class_color_{label}"
        )

        canvas_width = min(900, image_array.shape[1])
        scale = canvas_width / image_array.shape[1]
        canvas_height = int(image_array.shape[0] * scale)
        canvas_result = st_canvas(
            fill_color=_hex_to_rgba(selected_color, 0.35),
            stroke_width=2,
            stroke_color=selected_color,
            background_color="#ffffff",
            background_image=Image.fromarray(image_array).resize(
                (canvas_width, canvas_height)
            ),
            update_streamlit=False,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            key=f"canvas_{label}",
        )

        if st.button("Save drawn boxes", key=f"save_boxes_{label}"):
            boxes = _extract_boxes(canvas_result, scale, image_array.shape)
            if not boxes:
                st.warning("Draw at least one rectangle before saving.")
            else:
                additions = [
                    {"bbox": bbox, "category": selected_class, "color": selected_color}
                    for bbox in boxes
                ]
                st.session_state.manual_annotations[label].extend(additions)
                if selected_class not in st.session_state.class_names:
                    st.session_state.class_names.append(selected_class)
                if selected_class not in st.session_state.class_colors:
                    st.session_state.class_colors[selected_class] = selected_color
                st.success(f"Added {len(additions)} boxes for {selected_class}.")

        if st.session_state.manual_annotations[label]:
            st.caption("Saved manual annotations")
            st.dataframe(
                [
                    {
                        "Category": ann["category"],
                        "x1": int(ann["bbox"][0]),
                        "y1": int(ann["bbox"][1]),
                        "x2": int(ann["bbox"][2]),
                        "y2": int(ann["bbox"][3]),
                    }
                    for ann in st.session_state.manual_annotations[label]
                ],
                hide_index=True,
                use_container_width=True,
            )
            if st.button("Clear manual annotations", key=f"clear_{label}"):
                st.session_state.manual_annotations[label] = []
                st.info("Cleared manual annotations for this page.")


def _extract_boxes(canvas_result, scale, image_shape):
    if canvas_result is None or canvas_result.json_data is None:
        return []
    objects = canvas_result.json_data.get("objects", [])
    boxes = []
    for obj in objects:
        if obj.get("type") != "rect":
            continue
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        width = obj.get("width", 0) * obj.get("scaleX", 1)
        height = obj.get("height", 0) * obj.get("scaleY", 1)
        x1 = max(0, int(left / scale))
        y1 = max(0, int(top / scale))
        x2 = min(image_shape[1], int((left + width) / scale))
        y2 = min(image_shape[0], int((top + height) / scale))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def _hex_to_rgba(color: str, alpha: float):
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _render_download_button(label: str, image_array: np.ndarray):
    buffer = BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    st.download_button(
        label=f"Download coloured page ({label})",
        data=buffer.getvalue(),
        file_name=f"{label.replace(' ', '_')}_coloured.png",
        mime="image/png",
        key=f"download_{label}",
    )


def _render_pdf_download(pages: List[Tuple[str, np.ndarray]]):
    pdf_buffer = BytesIO()
    pil_images = [Image.fromarray(img).convert("RGB") for _, img in pages]
    if not pil_images:
        return
    first, *rest = pil_images
    first.save(pdf_buffer, format="PDF", save_all=True, append_images=rest)
    st.download_button(
        label="Download all coloured sheets (PDF)",
        data=pdf_buffer.getvalue(),
        file_name="coloured_sheets.pdf",
        mime="application/pdf",
        key="download_all_pdf",
    )
