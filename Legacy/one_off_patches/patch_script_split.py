import re

with open("Plans/PatchCore.md", "r") as f:
    content = f.read()

# 1. New Main Chart (Just high-level PatchCore pipeline)
new_section_1 = """## 1. Overall System Architecture Flow

The following graph illustrates the overall data pipeline and the high-level workflow for the PatchCore Few-Shot matching framework.

```mermaid
graph TD
    A[Raw PD Measurements Lab + Field] --> B[Tensor Generation]
    B --> C[128x3600 Phase-Resolved Matrices]
    
    C -->|Sliding Windows 128x256| F[Deep Learning Pathway]
    
    F -->|Few-Shot Memory Matching| I[PatchCore]
    I --> J[Extract Shallow ResNet-50 Features Layers 1 & 2]
    J --> K[Greedy Coreset Compression]
    K --> L[Memory Bank Construction]
    
    L --> O[K-NN Distance Search]
    
    O --> P[Performance Evaluation AU-ROC & PR-AUC]
    
    P --> Q[Final System Alarm Thresholds]
```"""

content = re.sub(
    r"## 1\. Overall System Architecture Flow\n\nThe following graph illustrates.*?\n```\n",
    new_section_1 + "\n",
    content,
    flags=re.DOTALL
)

# 2. Phase 1 Chart
phase_1_chart = """### Phase 1: Feature Extraction & Memory Bank Construction
During the training phase, the system builds a comprehensive dictionary of what "Normal" PRPD patches look like.

```mermaid
graph TD
    A[Normal Training Images Lab PD] --> B[ResNet-50 Pre-trained]
    
    subgraph Feature Extraction
        B -->|Severed Deep Layers| C[Extract Layer 1 & 2 Spatial Tensors]
        C --> D[Average Pooling & Concatenation]
        D --> E[Massive Raw Feature Pool e.g., 5.6M Features]
    end

    subgraph Greedy Coreset Compression
        E --> F{Is Feature Unique?}
        F -->|Yes: Maximizes Distance from existing core| G[Add to Coreset]
        F -->|No: Highly redundant e.g., empty space| H[Discard Feature]
        G --> I[Stop at 0.1% Compression Ratio]
    end

    subgraph Faiss Optimization
        I --> J[Subsampled Coreset e.g., 5k Features]
        J --> K[Build Faiss Index]
        K --> L[(Final Memory Bank)]
    end
```
"""

content = re.sub(
    r"### Phase 1: Feature Extraction & Memory Bank Construction\nDuring the training phase, the system builds a comprehensive dictionary of what \"Normal\" PRPD patches look like \(as detailed in the main flowchart above\)\.",
    phase_1_chart.strip(),
    content,
    flags=re.DOTALL
)

# 3. Phase 2 Chart
phase_2_chart = """### Phase 2: Anomaly Inference via K-NN Distance Search
During inference, a new unseen image is passed through the same ResNet backbone, and its features are compared mathematically against the saved Memory Bank.

```mermaid
graph TD
    A[Unseen Test Image] --> B[ResNet-50 Layer 1 & 2]
    B --> C[Test Feature Patches]
    
    subgraph K-NN Distance Search
        C --> D[Patch 1]
        C --> E[Patch 2]
        C --> F[Patch N]
        
        D --> G{Faiss K-Nearest Neighbor}
        E --> G
        F --> G
        
        H[(Faiss Memory Bank)] --> G
    end
    
    subgraph Scoring
        G --> I[Map distances back to Spatial Heatmap]
        I --> J[Apply Gaussian Smoothing]
        J --> K[Find Maximum Pixel Distance]
        K --> L[Final Image Anomaly Score]
    end
```
"""

content = re.sub(
    r"### Phase 2: Anomaly Inference via K-NN Distance Search\nDuring inference, a new unseen image is passed through the same ResNet backbone, and its features are compared mathematically against the saved Memory Bank \(as detailed in the main flowchart above\)\.",
    phase_2_chart.strip(),
    content,
    flags=re.DOTALL
)

with open("Plans/PatchCore.md", "w") as f:
    f.write(content)
print("Split successful.")
