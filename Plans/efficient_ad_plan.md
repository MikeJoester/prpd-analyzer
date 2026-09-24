# EfficientAD Implementation Plan for PRPD Anomaly Detection

## 1. Datasets & Custom Split Plan
This plan diverges from the baseline `ANOMALY.md` to incorporate real-world variance directly into the learning phase.
*   **Training Set:** `Lab PD` + `Field PD`
*   **Test and Evaluation Set:** `Field PD` only.
*   **Reasoning:** By including `Field PD` in the training set, the model learns the acceptable variance, attenuation, and noise floors of real-world measurements rather than overfitting strictly to pristine laboratory data. The evaluation is strictly restricted to unseen dates of `Field PD` to ensure the model generalizes to new operational conditions without data leakage.

## 2. Model Architecture & Step-by-Step Roles
We will employ a two-tiered approach: a classical Machine Learning baseline (SVM) and a deep-learning SOTA architecture (EfficientAD).

### 2.1 The Baseline Step: One-Class SVM (Support Vector Machine)
*   **Role:** Acts as the classical statistical baseline. It operates on the 256-dimensional hand-crafted mathematical features (the mean and max values extracted per phase bin).
*   **How it works:** The SVM plots the 256 features in a high-dimensional space and attempts to draw a tight mathematical boundary (a hypersphere) around the normal `Lab + Field PD` training points. During evaluation, if a new `Field PD` sample lands outside this boundary, it is flagged as anomalous.
*   **Parameters:** It uses an RBF (Radial Basis Function) kernel, which allows the boundary to be non-linear and flexibly wrap around complex data clusters.

### 2.2 The Deep Detector Step: EfficientAD (CNN-based)
EfficientAD is a state-of-the-art visual anomaly detector that processes the raw `128x256` PRPD image directly. It relies on the interaction of three distinct Convolutional Neural Networks (CNNs).

*   **The Teacher CNN:** A randomly initialized, completely frozen network.
    *   **Role:** It extracts "random" but mathematically deterministic features from the PRPD image. It acts as an arbitrary anchor point.
*   **The Student CNN:** An identical architecture to the Teacher.
    *   **Role:** It is actively trained to perfectly mimic the Teacher's output when fed the normal training data. When fed an anomaly during testing, the Student will fail to mimic the Teacher because it has never learned how to process those abnormal patterns.
*   **The Autoencoder CNN:** Contains a massive bottleneck (compressing the data down to 64 dimensions).
    *   **Role:** It learns to compress and reconstruct the global, logical structure of the PD pulses. If a valid PD pulse happens at the wrong phase angle (a "logical" anomaly), the Autoencoder will reconstruct it incorrectly because it violates the global rules it learned.

## 3. Loss Functions & Deep Dive into Values
During training, the EfficientAD network optimizes three distinct loss values simultaneously:

1.  **L_ST (Student-Teacher Loss):** 
    *   *Formula:* `MSE(Teacher_output, Student_output)`
    *   *Role:* Penalizes the Student for failing to match the Teacher. 
    *   *Deep Value Dive:* We use a **Hard Feature Loss** with a mining factor of p_hard = 0.999. This means we *only* calculate the loss on the worst 0.1% of errors. This forces the student to focus strictly on the hardest patches of the normal images, preventing it from generalizing its imitation skills to abnormal images.
2.  **L_AE (Autoencoder Loss):**
    *   *Formula:* `MSE(Teacher_output, Autoencoder_output)`
    *   *Role:* Trains the Autoencoder to mimic the Teacher. Because of its 64-dim bottleneck, it can't memorize the image; it is forced to learn the fundamental rules of PRPD shapes.
3.  **L_STAE (Student-Autoencoder Loss):**
    *   *Formula:* `MSE(Autoencoder_output, Student_output)`
    *   *Role:* The Student is trained to predict the Autoencoder's output *as well*. This prevents false positives: Autoencoders naturally produce slightly blurry reconstructions on normal images. By teaching the Student to anticipate this blurriness, we prevent the system from flagging standard blurriness as an anomaly.

## 4. Scoring Mechanism (Inference)
During the Test phase on unseen `Field PD`:
1.  **Local Anomaly Map:** `(Teacher - Student)^2`. Finds structural anomalies (e.g., weird localized noise spots or interference).
2.  **Global Anomaly Map:** `(Autoencoder - Student)^2`. Finds logical anomalies (e.g., valid PD pulses occurring at the wrong phase angles).
3.  **Final Score:** We average the Local and Global maps, and take the maximum pixel value as the final anomaly score for that PRPD file.
