# Vectorizer

High-Performance automated Vectorization and OCR Document processing agent.

This repository monitors the `/inputs/` directory for new raster images. It classifies, enhances, and vectorizes these images, automatically recognizing text elements, and generates a composite, searchable vector PDF formatted for standard 8.5x11 printing in the `/outputs/` directory.

## How to use

1. Upload your raster images (JPEG, PNG, WEBP, TIFF) into the `/inputs/` directory.
2. Go to the **Actions** tab in GitHub.
3. Select the **Vectorizer Pipeline** workflow.
4. Click **Run workflow** to trigger the processing.
5. Once the workflow is complete, you can download the generated PDFs and DXFs from the workflow run's artifacts.

## Disclaimer

"Personal Use" is assumed for copyrighted material, and users are responsible for the content they upload. Please ensure you have the appropriate rights for any documents you process through this pipeline.
