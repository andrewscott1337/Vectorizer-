# Agent Definition: Project Chimera (Image-to-Vector Pipeline)

**Role:** High-Performance automated Vectorization and OCR Document processing agent.

**Description:** This agent operates within a GitHub repository to monitor a specific input directory for new raster images. It classifies, enhances, and vectorizes these images, automatically recognizing text elements, and generates a composite, searchable vector PDF formatted for standard 8.5x11 printing.

---

### Phase 1: The Automation Mechanics (GitHub Actions)

This pipeline is designed for a public repository. Users contribute by uploading files, and administrators trigger the processing.

* **Input Trigger:** The pipeline must support `workflow_dispatch` (manual trigger). This gives the repo owner control over when the compute-intensive processing runs.
* **Directory Structure:**
  * `/inputs/`: The public drop-folder for images (JPEG, PNG, WEBP, TIFF).
  * `/outputs/`: Where the agent commits the final generated PDFs.
* **The Runner Environment:** We will utilize standard `ubuntu-latest` GitHub runners. The workflow must install the necessary non-Python system dependencies:
  * `tesseract-ocr` (for text recognition)
  * `potrace` (for robust vector tracing)
  * `poppler-utils` (for PDF assembly testing)

---

### Phase 2: The Core Processing Pipeline (The Python "Engine")

This script is the brain of the operation. It must execute the following steps sequentially for every file found in `/inputs/`.

#### Step 1: Ingest and Classification (Auto-Detection)
The first task is analyzing the image properties to determine the optimal pre-processing strategy.

* **Metric Analysis:** The script analyzes the image using OpenCV and Numpy. It calculates a "noise score" based on pixel variance and edge distribution.
* **The Decision:**
  * **High Variance (Photo):** Identifies photos of physical paper. Activates the Denoise & Thresholding pipeline to isolate black lines from noisy backgrounds.
  * **Low Variance (Screenshot/Digital Art):** Identifies sharp CAD screenshots or digital diagrams. Activates the Alias Suppression pipeline to smooth pixelated edges.

#### Step 2: Intelligent Upscaling
Before vectorizing, the image must be resized to improve tracing resolution.

* **Algorithm:** We will use OpenCV’s DNN (Deep Neural Network) module to run the `EDSR_x4` super-resolution model. It is computationally lightweight, fast, and generates superior results for line art and text compared to simple interpolation. It transforms a standard screenshot into a high-fidelity input.

#### Step 3: Dual-Track Processing (Text vs. Graphics)
The single biggest challenge in vectorization is text. A tracer tries to turn "A" into a shape, which looks blurry and isn't searchable. The Chimera Agent solves this by splitting the workload.

**Track A: OCR (The Text Parser)**
1. The agent uses **Tesseract OCR** (via `pytesseract`) to scan the upscaled image.
2. It extracts the text strings, bounding box coordinates, font size metadata, and confidence scores.
3. **Quality Control Logic:** The script must implement a filtering mechanism to ignore text blocks where the detected font size is too small or the OCR confidence score falls below a reliable threshold. This prevents illegible smudges, tiny artifacts, or complex linework from being misidentified as garbage text.

**Track B: Trace (The Graphics Parser)**
1. The agent receives the upscaled bitmap and applies **Adaptive Thresholding**. Instead of a global cutoff, this method analyzes small regions of the image to account for uneven lighting, shadows from page folds, and yellowed or aged paper. This ensures faint lines are preserved while background noise is dropped.
2. It uses **Potrace** (via `pypotrace`) to convert the optimized bitmap into mathematical, scalable vector paths (SVG).

#### Step 4: Final Assembly (PDF Compositing)
The final step is reassembling the two tracks into a seamless document.

* **PDF Engine:** We use ReportLab to generate the final 8.5x11 PDF.
* **Reconstruction:**
  1. The vector graphic generated in Track B is imported as the background layer.
  2. The valid text recognized in Track A is overlaid on the PDF at its exact (x,y) location using standard PDF fonts (e.g., Helvetica).
* **Result:** The final document is a perfectly crisp vector graphic with real, selectable, and searchable text perfectly aligned with the visual content.

---

### Phase 3: Deployment & Public Engagement

Since this repo is intended for the public, clear communication is required to ensure users know how to interact with it.

* **Documentation:** The README must clearly state that "Personal Use" is assumed for copyrighted material, and users are responsible for the content they upload.
* **Output Management:** The GitHub Action will automatically commit the generated PDFs and DXFs to the `/outputs/` directory upon successful completion.
