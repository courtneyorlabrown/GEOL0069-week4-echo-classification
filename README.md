# GEOL0069 – Week 4: Unsupervised Echo Classification

## Objective

Classify Sentinel-3 altimetry waveforms into **leads** and **sea ice** using unsupervised learning methods.  
Compute the mean echo shape and standard deviation for each class and evaluate performance against ESA official classification using a confusion matrix.

---

## Method

1. Load Sentinel-3 SAR altimetry waveform data (.nc file)
2. Preprocess waveforms (peakiness and SSD features)
3. Apply KMeans clustering (k = 2)
4. Assign clusters to physical classes (lead / sea ice)
5. Compute:
   - Mean waveform per class
   - Standard deviation per class
   - Confusion matrix vs ESA labels

---

## Repository Structure

- `notebooks/` – Google Colab notebook
- `outputs/` – Output figures (mean waveforms & confusion matrix)

---

## Results

Figures will be added after model execution:
- Mean echo shape ± standard deviation
- Confusion matrix comparison with ESA classification

