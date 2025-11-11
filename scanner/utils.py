from PIL import Image
import numpy as np
import os
import io
from typing import List, Tuple
from pdf2image import convert_from_path, convert_from_bytes


def load_test_image():
    file_name = 'Poker Face Fl 2.pdf'
    # file_name = 'lg-2267728-aug-beethoven--page-2.png' 
    test_im_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test_data', file_name)
    img_file = convert_from_path(test_im_path)[0]
    test_image = np.array(img_file)
    return test_image

def get_photo(file, key):
    images, _ = load_uploaded_file(file)
    if not images:
        raise ValueError("Unable to decode uploaded file.")
    # Preserve previous behaviour by returning only the first page/image
    return images[0]


def load_uploaded_file(file) -> Tuple[List[np.ndarray], List[str]]:
    """Convert an uploaded (potentially multi-page) file into numpy images.

    Returns:
        Tuple[List[np.ndarray], List[str]]: list of images and human readable labels
            (useful for Streamlit display when a PDF yields multiple pages).
    """
    bytes_data = file.getvalue()
    file_suffix = os.path.splitext(file.name)[1].lower()
    mime_subtype = file.type.split('/')[-1].lower()

    if file_suffix == '.pdf' or mime_subtype == 'pdf':
        pil_images = convert_from_bytes(bytes_data)
        labels = [f"{file.name} - page {idx + 1}" for idx in range(len(pil_images))]
    else:
        pil_images = [Image.open(io.BytesIO(bytes_data))]
        labels = [file.name]

    numpy_images = [np.array(img.convert("RGB")) for img in pil_images]
    return numpy_images, labels
