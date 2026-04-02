import cv2
from cv2 import dnn_superres
import numpy as np
import pytesseract
import potrace
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import os
import glob
import xml.etree.ElementTree as ET
import tempfile
import ezdxf

BLOCK_SIZE = 21
C_VAL = 10

def classify_and_preprocess(image_path):
    print(f"Processing {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate noise score based on Laplacian variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Image variance: {variance}")

    if variance > 500: # Threshold for high variance (photo) vs low variance (screenshot)
        print("Classification: High Variance (Photo)")
        # Smooth out paper grain before thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Denoise & Thresholding
        denoised = cv2.fastNlMeansDenoising(blurred, None, 10, 7, 21)
        # Using adaptive thresholding for better preservation of lines
        processed = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C_VAL)
    else:
        print("Classification: Low Variance (Screenshot/Digital Art)")
        # Alias Suppression (smooth pixelated edges)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(blurred, -1, kernel)

    return img, processed

if __name__ == "__main__":
    pass

def upscale_image(img, model_path="EDSR_x4.pb"):
    print("Upscaling image...")
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4)
    upscaled = sr.upsample(img)
    return upscaled

def extract_text(image):
    print("Track A: Extracting text (OCR)...")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    valid_text = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60: # Confidence threshold
            text = data['text'][i].strip()
            if text:
                # Basic check for tiny text/artifacts based on height
                if data['height'][i] > 10:
                    valid_text.append({
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i],
                        'conf': data['conf'][i]
                    })
    return valid_text

def create_svg(image, output_svg_path):
    print("Track B: Tracing graphics...")

    # Needs to be binary for potrace
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Adaptive Thresholding again on upscaled image for best results
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C_VAL)

    # pypotrace requires boolean numpy array, where True is black (foreground)
    # usually threshold makes background 255 (white) and foreground 0 (black)
    bitmap = binary == 0

    path = potrace.Bitmap(bitmap).trace()

    h, w = bitmap.shape

    with open(output_svg_path, "w") as fp:
        fp.write(f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n')
        fp.write('<path d="')
        for curve in path:
            fp.write(f"M {curve.start_point[0]},{h - curve.start_point[1]} ")
            for segment in curve.segments:
                if segment.is_corner:
                    fp.write(f"L {segment.c[0]},{h - segment.c[1]} L {segment.end_point[0]},{h - segment.end_point[1]} ")
                else:
                    fp.write(f"C {segment.c1[0]},{h - segment.c1[1]} {segment.c2[0]},{h - segment.c2[1]} {segment.end_point[0]},{h - segment.end_point[1]} ")
        fp.write('" fill="black" stroke="none" />\n')
        fp.write('</svg>')

    return path

def create_dxf(path, output_dxf_path, h):
    print("Track B: Generating DXF...")
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    for curve in path:
        start_pt = curve.start_point
        for segment in curve.segments:
            if segment.is_corner:
                msp.add_line((segment.c[0], h - segment.c[1]), (segment.end_point[0], h - segment.end_point[1]))
                start_pt = segment.end_point
            else:
                msp.add_spline(fit_points=[
                    (start_pt[0], h - start_pt[1]),
                    (segment.c1[0], h - segment.c1[1]),
                    (segment.c2[0], h - segment.c2[1]),
                    (segment.end_point[0], h - segment.end_point[1])
                ])
                start_pt = segment.end_point

    doc.saveas(output_dxf_path)


def assemble_pdf(svg_path, text_data, output_pdf_path, img_width, img_height):
    print("Step 4: Assembling PDF...")
    # Standard 8.5x11 inches in points
    pdf_width, pdf_height = letter

    c = canvas.Canvas(output_pdf_path, pagesize=letter)

    # Calculate scaling to fit the image on the page
    scale_x = pdf_width / img_width
    scale_y = pdf_height / img_height
    scale = min(scale_x, scale_y)

    # Center offsets
    offset_x = (pdf_width - (img_width * scale)) / 2
    offset_y = (pdf_height - (img_height * scale)) / 2

    # Draw SVG Background
    drawing = svg2rlg(svg_path)
    if drawing:
        drawing.scale(scale, scale)
        renderPDF.draw(drawing, c, offset_x, offset_y)

    # Overlay Text
    c.setFont("Helvetica", 10) # default font
    c.setFillColorRGB(0, 0, 0, alpha=0) # Make text invisible but selectable

    for t in text_data:
        # Convert coords based on scale
        # PDF origin is bottom-left, image origin is top-left
        # Also adjust font size relative to scale and height

        pdf_x = offset_x + (t['x'] * scale)
        pdf_y = offset_y + ((img_height - t['y'] - t['h']) * scale) # flip y

        font_size = t['h'] * scale * 0.8 # rough approximation
        c.setFont("Helvetica", max(4, font_size))

        c.drawString(pdf_x, pdf_y, t['text'])

    c.save()

def process_file(input_file, output_dir):
    filename = os.path.basename(input_file)
    name, _ = os.path.splitext(filename)
    output_pdf = os.path.join(output_dir, f"{name}.pdf")
    output_dxf = os.path.join(output_dir, f"{name}.dxf")

    try:
        # Step 1: Classification & Preprocessing
        original_img, processed_img = classify_and_preprocess(input_file)

        # Step 2: Upscale
        # EDSR expects a 3-channel BGR image
        if len(processed_img.shape) == 2:
            img_for_upscaling = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        else:
            img_for_upscaling = processed_img

        upscaled_img = upscale_image(img_for_upscaling)

        # Track A: OCR
        # Tesseract performs better on upscaled image
        text_data = extract_text(upscaled_img)

        # Track B: Trace
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
            svg_path = tmp_svg.name

        path = create_svg(upscaled_img, svg_path)

        # Step 4: Assemble PDF & DXF
        h, w = upscaled_img.shape[:2]
        assemble_pdf(svg_path, text_data, output_pdf, w, h)
        create_dxf(path, output_dxf, h)

        # Clean up temp svg
        os.remove(svg_path)
        print(f"Successfully created: {output_pdf} and {output_dxf}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    input_dir = "inputs"
    output_dir = "outputs"

    os.makedirs(output_dir, exist_ok=True)

    # Look for common image formats
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.tiff")
    files_to_process = []

    for ext in valid_extensions:
        files_to_process.extend(glob.glob(os.path.join(input_dir, ext)))
        files_to_process.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    print(f"Found {len(files_to_process)} files to process.")

    for f in files_to_process:
        process_file(f, output_dir)
