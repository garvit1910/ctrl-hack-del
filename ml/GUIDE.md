# TremorTrace ML Pipeline — Next Steps Guide

## Current Status

All source code is complete:

```
ml/
├── src/                              ✅ 8 modules + __init__.py
│   ├── config.py                     ✅ Hyperparameters & constants
│   ├── data_pipeline.py              ✅ Augmentation & data generators
│   ├── model_builder.py              ✅ MobileNetV2 model construction
│   ├── trainer.py                    ✅ Two-phase transfer learning
│   ├── evaluator.py                  ✅ 7 visualization types
│   ├── gradcam.py                    ✅ Grad-CAM explainability
│   ├── ensemble.py                   ✅ 2-model weighted ensemble
│   └── input_handler.py              ✅ Backend entry point
├── notebooks/
│   └── TremorTrace_Training.ipynb    ✅ 19-cell Colab notebook
├── exports/
│   └── README.md                     ✅ Backend integration docs
└── requirements.txt                  ✅ Dependencies
```

**Models are NOT trained yet.** You need a GPU (Google Colab T4) to train them.

---

## Step 1: Get a Kaggle API Key

The dataset is hosted on Kaggle. You need credentials to download it.

1. Go to https://www.kaggle.com — sign up or log in
2. Click your profile icon (top right) → **Settings**
3. Scroll to **API** section → click **"Create New Token"**
4. It downloads a `kaggle.json` file containing your username and key
5. Keep these ready — Cell 2 of the notebook will ask for them

---

## Step 2: Upload Source Files to Google Drive

Create this exact folder structure in your Google Drive:

```
My Drive/
  TremorTrace/
    src/
      __init__.py
      config.py
      data_pipeline.py
      model_builder.py
      trainer.py
      evaluator.py
      gradcam.py
      ensemble.py
      input_handler.py
```

**How to upload:**
1. Open Google Drive (drive.google.com)
2. Create folder: `TremorTrace`
3. Inside it, create folder: `src`
4. Upload all 9 files from your local `ml/src/` into `TremorTrace/src/`

---

## Step 3: Open the Notebook in Google Colab

1. Go to https://colab.research.google.com
2. Click **File → Upload notebook**
3. Upload `ml/notebooks/TremorTrace_Training.ipynb`
4. **CRITICAL: Enable GPU**
   - Go to **Runtime → Change runtime type**
   - Set **Hardware accelerator** to **T4 GPU**
   - Click **Save**

---

## Step 4: Run the Notebook (Cell by Cell)

Run each cell in order by pressing **Shift+Enter** or clicking the play button.

### Cell 1: Environment Setup (~30 seconds)
- Mounts your Google Drive
- Installs dependencies
- Verifies GPU is available
- **What to check:** You should see `✅ GPU available: ...`
- **If no GPU:** Go back to Runtime → Change runtime type → T4 GPU

### Cell 2: Download Dataset (~1 minute)
- Downloads the Parkinson's Drawings dataset from Kaggle
- **You will be prompted** for your Kaggle username and API key
- Enter them from the `kaggle.json` you downloaded in Step 1
- **What to check:** `✅ Dataset downloaded to: /content/parkinsons-drawings`

### Cell 3: Explore Dataset (instant)
- Counts images per class per split
- **What to check:** A table showing ~36 training images and ~15 test images per class

### Cell 4: Visualize Samples (instant)
- Shows a 4×5 grid of sample images
- **What to check:** You should see spirals and waves from both healthy and PD patients

### Cell 5: Create Data Generators (~5 seconds)
- Sets up train/validation/test data pipelines
- **What to check:** `✅ Spiral generators created` and `✅ Wave generators created`

### Cell 6: Verify Augmentation (instant)
- Shows augmented versions of images
- **What to check:** Spirals should show full rotations; waves should show slight rotations only

### Cell 7: Build Models (~10 seconds)
- Constructs both MobileNetV2 models
- **What to check:** Model summary printed, ~2.5M total params, ~200K trainable params

### Cell 8: Train Spiral CNN (~5-10 minutes)
- Two-phase training: frozen base → fine-tune
- **What to watch:** Loss decreasing, AUC increasing over epochs
- **What to check:** `✅ Spiral Phase 1 complete` and `✅ Spiral Phase 2 complete`

### Cell 9: Train Wave CNN (~5-10 minutes)
- Same process for the wave model
- **What to check:** `✅ Wave Phase 1 complete` and `✅ Wave Phase 2 complete`

### Cell 10: Training History Plots (instant)
- Shows loss, accuracy, and AUC curves over epochs
- **What to check:** Curves should trend in the right direction (loss down, accuracy/AUC up)
- The dashed green line shows where Phase 2 (fine-tuning) started

### Cell 11: Evaluate Both Models (instant) ← THIS IS WHERE YOU SEE ACCURACY
- Prints classification reports with precision, recall, F1
- Shows confusion matrices and ROC curves
- **What to check:** The final printout:
  ```
  ✅ Spiral CNN — Accuracy: XX%, AUC: X.XXXX
  ✅ Wave CNN   — Accuracy: XX%, AUC: X.XXXX
  ```

### Cell 12: Sample Predictions (instant)
- 2×5 grid of test images with predicted percentages and risk tiers
- Green title = correct prediction, Red = incorrect

### Cell 13: Grad-CAM Visualization (~30 seconds)
- Heatmaps showing what each model focuses on in the drawings
- **What to check:** Heatmaps should highlight the drawn lines, not the background

### Cell 14: Test Ensemble (instant)
- Runs the 2-model ensemble on test samples
- Shows a table: spiral %, wave %, ensemble %, risk tier, agreement

### Cell 15: Risk Tier Distribution (instant)
- Bar charts showing how predictions spread across risk tiers
- **What to check:** Healthy samples should cluster in Low/Mild tiers, PD in Elevated/High

### Cell 16: Model Comparison (instant)
- Side-by-side bar chart: Spiral CNN vs Wave CNN

### Cell 17: Test Input Handler (instant)
- End-to-end test of `process_input()` — the function your backend will call
- **What to check:** Both "drawn" and "uploaded" modes produce identical predictions

### Cell 18: Export Package (~10 seconds)
- Copies trained models and backend files to `TremorTrace/exports/` in your Drive
- **What to check:** All files listed with ✅

### Cell 19: Final Dashboard (instant)
- Summary of everything: metrics, Grad-CAM samples, architecture overview
- `✅ TremorTrace ML Pipeline Complete!`

---

## Step 5: Download the Exports

After the notebook finishes:

1. Open Google Drive → `TremorTrace/exports/`
2. Download these files:
   - `spiral_final.keras` — trained spiral model
   - `wave_final.keras` — trained wave model
   - `ensemble.py` — ensemble logic
   - `gradcam.py` — Grad-CAM generation
   - `input_handler.py` — backend entry point
   - `metadata.json` — model metrics and config

3. Place them in your project's backend directory

---

## Step 6: Backend Integration

Your backend only needs to do this:

```python
import tensorflow as tf
from input_handler import process_input

# Load models ONCE at server startup
spiral_model = tf.keras.models.load_model('path/to/spiral_final.keras')
wave_model = tf.keras.models.load_model('path/to/wave_final.keras')

# For each request:
result = process_input(
    spiral_image_base64=spiral_b64,    # base64 PNG from frontend
    wave_image_base64=wave_b64,        # base64 PNG from frontend
    spiral_cnn_model=spiral_model,
    wave_cnn_model=wave_model,
    input_mode='drawn',                # or 'uploaded'
)

# result['pd_probability_percent'] → 0.0 to 100.0
# result['risk_tier'] → "Low Risk", "Mild Risk", etc.
# result['risk_color'] → hex color for UI
# result['spiral_gradcam_base64'] → heatmap image for display
# result['wave_gradcam_base64'] → heatmap image for display
```

---

## Expected Results

With ~72 training images per type (this is tiny), expect:

| Outcome | Accuracy | AUC | Notes |
|---------|----------|-----|-------|
| **Great** | 85-95% | 0.90-0.98 | Transfer learning working well |
| **Good** | 75-85% | 0.80-0.90 | Typical for this dataset size |
| **Okay** | 65-75% | 0.70-0.80 | Model may need more augmentation |

Results vary between runs due to the small dataset. This is expected for a hackathon project.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No GPU detected" | Runtime → Change runtime type → T4 GPU |
| Kaggle download fails | Double-check username/key, or download dataset manually from kaggle.com and upload to `/content/parkinsons-drawings/` |
| `ModuleNotFoundError: No module named 'src'` | Make sure all src/ files are in `Drive/TremorTrace/src/` and Cell 1 ran successfully |
| Out of GPU memory | Runtime → Restart runtime → Rerun from Cell 1 |
| Training stuck / very slow | Make sure you selected T4 GPU, not CPU |
| Low accuracy (<65%) | Try rerunning training (Cells 8-9) — random initialization varies |
| `src/ not found` warning in Cell 1 | Upload src/ files to Google Drive path shown in the error |

---

## Timeline Summary

| Phase | Time |
|-------|------|
| Setup (Cells 1-7) | ~3 minutes |
| Training (Cells 8-9) | ~10-20 minutes |
| Evaluation (Cells 10-17) | ~2 minutes |
| Export (Cells 18-19) | ~1 minute |
| **Total** | **~20-25 minutes** |
