# GEOL0069 – Week 4: Unsupervised Echo Classification

## Objective

Classify Sentinel-3 SAR altimetry waveforms into **leads** and **sea ice** using unsupervised learning.

Compute:

- Mean echo shape per class  
- Standard deviation per class  
- Confusion matrix vs ESA official classification  

---

## Method

1. Load Sentinel-3 SAR waveform data (`.nc` file)
2. Extract waveform features (e.g., peakiness and SSD)
3. Apply KMeans clustering (`k = 2`)
4. Assign clusters to physical classes (lead / sea ice)
5. Evaluate against ESA labels using a confusion matrix

---

## Repository Structure

notebooks/ → Google Colab notebook
outputs/ → Generated figures
README.md → Documentation and interpretation
Requirements → Python package requirements

---

# Results & Interpretation

The notebook produces two required outputs:

1. **Mean echo shape ± standard deviation**
2. **Confusion matrix vs ESA classification**

All figures are saved inside the `outputs/` folder.

---

## Output 1 — Mean Echo Shape ± Standard Deviation

![Mean Echo Shapes](outputs/1_mean_echo_shapes.png)

### Interpretation

Each Sentinel-3 waveform is a 1D radar return sampled across **range bins**.

A **range bin** represents one discrete sampling position along the radar’s range window (distance from the satellite along the radar line-of-sight).

Waveform shape reflects surface properties:

- **Lead (open water)** → Narrow, sharp, high-amplitude peak (specular reflection)
- **Sea ice** → Broader, smoother peak (diffuse scattering)

In the figure:

- Solid lines = Mean waveform per class  
- Shaded region = ±1 standard deviation  

The sharper peak for the Lead class confirms physically distinct scattering behaviour compared to sea ice.

---

### Code Used to Generate Output 1

```python
import numpy as np
import matplotlib.pyplot as plt

# waves_cleaned: array of shape (n_waveforms, n_bins)
# clusters_gmm: predicted cluster labels (0/1)

plt.figure(figsize=(8, 6))
xr = np.arange(waves_cleaned.shape[1])

for cls, name, color in [(0, "Sea ice", "green"),
                         (1, "Lead", "blue")]:
    
    mask = (clusters_gmm == cls)

    mean_wave = np.nanmean(waves_cleaned[mask], axis=0)
    std_wave  = np.nanstd(waves_cleaned[mask], axis=0)

    plt.plot(xr, mean_wave, color=color, label=name)
    plt.fill_between(
        xr,
        mean_wave - std_wave,
        mean_wave + std_wave,
        color=color,
        alpha=0.3
    )

plt.xlabel("Range Bin")
plt.ylabel("Power")
plt.title("Mean Echo Shape ± Standard Deviation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("outputs/1_mean_echo_shapes.png",
            dpi=200,
            bbox_inches="tight")

plt.show()
```
## Output 2 — Confusion Matrix vs ESA Classification

![Confusion Matrix](outputs/2_confusion_matrix.png)

### Interpretation

This matrix compares our unsupervised clustering against ESA's official surface classification.

- Rows = True ESA labels

- Columns = Predicted labels

- Diagonal values = Correct classifications

- Off-diagonal values = Misclassifications

A strong diagonal indicates strong agreement with ESA labels, confirming that waveform-based clustering successfully separates leads and sea ice.

### Code Used to Generate Output 2
```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# y_true: ESA labels (0/1)
# y_pred: predicted labels from clustering

conf_matrix = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=["Sea ice", "Lead"]
)

disp.plot(values_format="d")

plt.title("Confusion Matrix vs ESA Classification")
plt.tight_layout()

plt.savefig("outputs/2_confusion_matrix.png",
            dpi=200,
            bbox_inches="tight")

plt.show()
```

## How to Reproduce

1. Open the notebook in notebooks/ using Google Colab.

2. Ensure the .nc file path points to your dataset location.

3. Run all cells from top to bottom.

4. The following figures will be generated:

outputs/1_mean_echo_shapes.png
outputs/2_confusion_matrix.png

5. Commit and push the updated notebook and outputs folder to GitHub.

## Notes on Variable Names

- Depending on notebook version, variable names may differ:

- Waveform array → waves, waves_cleaned, or waves_norm

- Cluster labels → clusters_kmeans or clusters_gmm

- ESA labels → ESA_labels, surf_type, or y_true

## To inspect variables in your environment:

```
print([v for v in dir() if "wave" in v.lower()])
print([v for v in dir() if "cluster" in v.lower()])
```
