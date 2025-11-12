from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import PredictionResult, ObjectPrediction
from sahi.postprocess.combine import GreedyNMMPostprocess
from sahi.utils.cv import visualize_object_predictions
from sahi.annotation import BoundingBox
from scanner.yolo10_sahi_detection_model import Yolov10DetectionModel
import torch
import numpy as np
import os
from functools import cmp_to_key
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights/best.pt')

detection_model = Yolov10DetectionModel(
    model_path=model_weights,
    confidence_threshold=0.5,
    device="cuda:0", # 'cpu' or 'cuda:0'
)

def detect_everything(
    source: np.array
):
    """detect objects

    Args:
        source (np.array): image 

    Returns:
        PredictionResult: image and prediction_list
    """
    return get_sliced_prediction(
        source,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_type="NMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.1,
    )


def debug_tile_predictions(
    source: np.ndarray,
    tile_size: int = 640,
    overlap_ratio: float = 0.2,
) -> List[Dict[str, np.ndarray]]:
    tiles = _generate_tiles(source, tile_size, overlap_ratio)
    per_tile = []
    img_h, img_w = source.shape[:2]
    for tile_image, (offset_x, offset_y) in tiles:
        tile_result = get_prediction(
            tile_image,
            detection_model,
            shift_amount=[0, 0],
            full_shape=[tile_image.shape[0], tile_image.shape[1]],
        )
        per_tile.append({
            "image": tile_image,
            "predictions": tile_result.object_prediction_list,
        })
    return per_tile

DEFAULT_COLOR_PALETTE = [
    "#F94144",
    "#F8961E",
    "#F9C74F",
    "#90BE6D",
    "#43AA8B",
    "#577590",
    "#277DA1",
    "#9B5DE5",
    "#F15BB5",
    "#00BBF9",
]


def visualize_predictions(
    image: np.array,
    predictions_list,
    filter=None,
    text_size: float = None,
    rect_th: int = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
    class_colors: Optional[Dict[str, str]] = None,
    fill_alpha: float = 0.25,
):
    """visualize image 

    Args:
        image (np.array): image with detected objects
        predictions_list (_type_): list of detected objects
        filter (_type_, optional): what classes to visualize. Defaults to None.
        text_size (float, optional): size of the category name over box. Defaults to None.
        rect_th (int, optional): rectangle thickness. Defaults to None.
        hide_labels (bool, optional): hide labels. Defaults to False.
        hide_conf (bool, optional): hide confidence. Defaults to False.

    Returns:
        np.array: image with bboxes
    """
    if filter:
        predictions_list = filter_predictions(predictions_list, filter)

    if class_colors:
        return _visualize_with_custom_colors(
            image,
            predictions_list,
            class_colors,
            text_size=text_size,
            rect_th=rect_th,
            hide_labels=hide_labels,
            hide_conf=hide_conf,
            fill_alpha=fill_alpha,
        )

    im_with_det = visualize_object_predictions(
        image=np.ascontiguousarray(image),
        object_prediction_list=predictions_list,
        rect_th=rect_th,
        text_size=text_size,
        text_th=None,
        color=None,
        hide_labels=hide_labels,
        hide_conf=hide_conf,
    )

    return im_with_det["image"]

def filter_predictions(prediction_list, classes_to_view):
    filtered_list = []
    for prediction in prediction_list:
        if prediction.category.name in classes_to_view:
            filtered_list.append(prediction)
    return filtered_list

def compare_boxes_vertically(obj1:ObjectPrediction, obj2:ObjectPrediction):
    return obj1.bbox.to_xyxy()[1] - obj2.bbox.to_xyxy()[1]

def compare_boxes_horizontally(obj1:ObjectPrediction, obj2:ObjectPrediction):
    return obj1.bbox.to_xyxy()[0] - obj2.bbox.to_xyxy()[0]

def get_mid_lines(sorted_staffs:List[ObjectPrediction]):
    lines = []
    for i in range(len(sorted_staffs)-1):
        lines.append((sorted_staffs[i].bbox.to_xyxy()[3] + sorted_staffs[i+1].bbox.to_xyxy()[1]) / 2)
    return lines

def get_slice_bbox(lines, image_size):
    w, h = image_size
    y_min = 0
    # [x_min, y_min, x_max, y_max]
    slice_bboxes = []
    for line in lines:
        slice_bboxes.append([0, int(y_min), w, int(line)])
        y_min = int(line)
    slice_bboxes.append([0, y_min, w, h])
    return slice_bboxes

def prediction_inside_slice(prediction: ObjectPrediction, slice_bbox: List[int]) -> bool:
    """Check whether prediction coordinates lie inside slice coordinates.

    Args:
        prediction (dict): Single prediction entry in COCO format.
        slice_bbox (List[int]): Generated from `get_slice_bbox`.
            Format for each slice bbox: [x_min, y_min, x_max, y_max].

    Returns:
        (bool): True if any annotation coordinate lies inside slice.
    """
    left, top, right, bottom = prediction.bbox.to_xyxy()

    if right <= slice_bbox[0]:
        return False
    if bottom <= slice_bbox[1]:
        return False
    if left >= slice_bbox[2]:
        return False
    if top >= slice_bbox[3]:
        return False
    
    return True

def transform_prediction(prediction: ObjectPrediction, slice_bbox: List[int]):
    # prediction.bbox.miny = prediction.bbox.miny - slice_bbox[1]
    # prediction.bbox.maxy = prediction.bbox.maxy - slice_bbox[1]    
    bbox = prediction.bbox
    # construct a new bbox object
    shift_y = slice_bbox[1]
    miny = bbox.miny - shift_y
    maxy = bbox.maxy - shift_y
    # guard against negative coordinates introduced by slicing near the image edge
    miny = max(miny, 0)
    maxy = max(maxy, miny)
    new_bbox = BoundingBox(
        [bbox.minx,
        miny,
        bbox.maxx,
        maxy]
    )
    prediction.bbox = new_bbox


def postprocess(data, match_threshold=0.1, match_metric="IOU", class_agnostic=False):
    postprocess = GreedyNMMPostprocess(
        match_threshold=match_threshold,
        match_metric=match_metric,
        class_agnostic=class_agnostic,
    )
    return postprocess(data)

def slice_image(predicted_image: PredictionResult, divider:str):
    # staffs will be sorted later
    sorted_staffs = filter_predictions(predicted_image.object_prediction_list, set([divider]))

    # merge staffs if there are intersections
    # postprocess = GreedyNMMPostprocess(
    #     match_threshold=0.1,
    #     match_metric="IOS",
    #     class_agnostic=False,
    # )

    # sort vertically
    if len(sorted_staffs) > 1:
        sorted_staffs = sorted(postprocess(sorted_staffs, match_metric="IOS"), key=cmp_to_key(compare_boxes_vertically))

    # get slice bboxes each with only one staff
    lines = get_mid_lines(sorted_staffs)
    slices = get_slice_bbox(lines, predicted_image.image.size)

    sliced_images = []
    for slice in slices:
        # extract image
        tlx = slice[0]
        tly = slice[1]
        brx = slice[2]
        bry = slice[3]
        slice_objects = []
        for object in predicted_image.object_prediction_list:
            # transform prediction coordinates if it is in the slice
            if prediction_inside_slice(object, slice):
                transform_prediction(object, slice)
                slice_objects.append(object)
        sliced_images.append({'image': np.asarray(predicted_image.image)[tly:bry, tlx:brx], 'predictions': slice_objects})
    return sliced_images


def _generate_tiles(
    image: np.ndarray, tile_size: int, overlap_ratio: float
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    height, width = image.shape[:2]
    stride = int(tile_size * (1 - overlap_ratio))
    stride = max(1, stride)
    x_positions = _compute_positions(width, tile_size, stride)
    y_positions = _compute_positions(height, tile_size, stride)
    tiles = []
    for top in y_positions:
        bottom = min(top + tile_size, height)
        for left in x_positions:
            right = min(left + tile_size, width)
            tiles.append((image[top:bottom, left:right], (left, top)))
    return tiles


def _compute_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if positions[-1] != last_start:
        positions.append(last_start)
    return sorted(set(max(0, min(pos, last_start)) for pos in positions))


def _visualize_with_custom_colors(
    image: np.ndarray,
    prediction_list,
    class_colors: Dict[str, str],
    text_size: Optional[float],
    rect_th: Optional[int],
    hide_labels: bool,
    hide_conf: bool,
    fill_alpha: float,
):
    if not prediction_list:
        return image

    base_image = Image.fromarray(np.ascontiguousarray(image)).convert("RGBA")
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    outline_width = rect_th if rect_th and rect_th > 0 else 0

    for prediction in prediction_list:
        color = _color_for_class(prediction.category.name, class_colors)
        x1, y1, x2, y2 = map(int, prediction.bbox.to_xyxy())
        fill = (*color, int(255 * np.clip(fill_alpha, 0, 1)))
        overlay_draw.rectangle([x1, y1, x2, y2], fill=fill)

    composed = Image.alpha_composite(base_image, overlay)
    draw = ImageDraw.Draw(composed)
    font = _resolve_font(composed.size, text_size)

    for prediction in prediction_list:
        color = _color_for_class(prediction.category.name, class_colors)
        x1, y1, x2, y2 = map(int, prediction.bbox.to_xyxy())
        if outline_width > 0:
            draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=outline_width)

        if not hide_labels:
            label = prediction.category.name
            if not hide_conf:
                label = f"{label} {prediction.score.value:.2f}"
            text_w, text_h = _measure_text(draw, label, font)
            text_x1 = x1
            text_y1 = max(y1 - text_h - 4, 0)
            text_bg = [text_x1, text_y1, text_x1 + text_w + 4, text_y1 + text_h + 4]
            draw.rectangle(text_bg, fill=(*color, 220))
            draw.text(
                (text_x1 + 2, text_y1 + 2),
                label,
                fill=(255, 255, 255, 255),
                font=font,
            )

    return np.array(composed.convert("RGB"))


def _color_for_class(class_name: str, class_colors: Dict[str, str]):
    if class_colors and class_name in class_colors:
        return _hex_to_rgb(class_colors[class_name])
    palette_index = abs(hash(class_name)) % len(DEFAULT_COLOR_PALETTE)
    return _hex_to_rgb(DEFAULT_COLOR_PALETTE[palette_index])


def _hex_to_rgb(value: str):
    if isinstance(value, (tuple, list)):
        if len(value) >= 3:
            return tuple(int(v) for v in value[:3])
    if not isinstance(value, str):
        return (255, 255, 255)

    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return (255, 255, 255)
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (r, g, b)


def _resolve_font(image_size, text_size: Optional[float]):
    base = int(min(image_size) / 30)
    if text_size:
        base = max(int(20 * text_size), 10)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=base)
    except IOError:
        return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    return draw.textsize(text, font=font)
