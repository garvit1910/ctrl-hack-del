# NeuroSketch

NeuroSketch is a clinical-grade neurological screening platform designed to detect early indicators of Parkinson’s Disease through hand-drawn motor analysis. By utilizing validated screening methods—specifically the Archimedes spiral and wave tests—the platform provides real-time results directly in the browser with zero data collection.

## Technical Overview

The project is architected as a high-performance web application, leveraging a deep learning ensemble to analyze motor tremors and drawing patterns.

### Frontend
The frontend is built with **Next.js 14** and **TypeScript**, focusing on an immersive 3D user experience and technical transparency.

* **UI/UX**: Implemented using **Tailwind CSS** and **Shadcn UI** for a clean, professional aesthetic.
* **3D Engine**: Features a custom 3D neural mesh with a first-person navigable camera, allowing users to explore the model's internal nodes.
* **Animations**: Utilizes **Framer Motion** for cinematic UI transitions, including proximity-based fact reveals and a state-driven cinematic mode.
* **Privacy**: Architected for zero-retention; data is processed and results are delivered without permanent storage.

### Backend
The backend serves as the inference engine, built with **FastAPI** for high-concurrency performance and **TensorFlow** for model execution.

* **Architecture**: Utilizes a **MobileNetV2** backbone pre-trained on ImageNet for efficient feature extraction, with custom dense layers optimized for multi-class classification.
* **Ensemble Logic**: Implements a weighted voting ensemble (40% Spiral / 60% Wave) to maximize diagnostic sensitivity.
* **Inference**: Processes image blobs and [x, y, t] coordinate data to evaluate tremor frequency and drawing accuracy.

## Model Performance

The current iteration of the system demonstrates high reliability across both primary screening modalities:

| Test Type | Accuracy | AUC |
| :--- | :--- | :--- |
| Spiral CNN | 86.7% | 0.9511 |
| Wave CNN | 91.3% | 0.9627 |

## Research and Data

The models were trained on the [Parkinson's Drawings dataset via Kaggle](https://www.kaggle.com/datasets/kmader/parkinsons-drawings). The methodology is informed by clinical research regarding the efficacy of hand-drawn Archimedes spirals as digital biomarkers for neurodegenerative conditions.

---
