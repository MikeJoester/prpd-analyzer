import re

with open("Plans/PatchCore.md", "r") as f:
    content = f.read()

# 1. Replace the first section
new_section_1 = """## 1. Overall System Architecture Flow

The following graph illustrates the data pipeline and the complete functional workflow for the PatchCore Few-Shot matching framework, detailing both the Training (Memory Bank Construction) and Inference (K-NN Scoring) phases.

```mermaid
graph TD
    %% Main Data Ingestion
    A[Raw PD Measurements Lab + Field] --> B[Tensor Generation]
    B --> C[128x3600 Phase-Resolved Matrices]
    C -->|Sliding Windows 128x256| D[PatchCore Deep Learning Pathway]

    %% Split into Training vs Inference
    D --> E{Mode}

    %% Training Pathway
    E -->|Training Phase Lab PD| F[ResNet-50 Pre-trained Backbone]
    
    subgraph Phase 1: Feature Extraction & Memory Bank Construction
        F -->|Severed Deep Layers| G[Extract Layer 1 & 2 Spatial Tensors]
        G --> H[Average Pooling & Concatenation]
        H --> I[Massive Raw Feature Pool e.g., 5.6M Features]

        %% Greedy Coreset
        subgraph Greedy Coreset Compression
            I --> J{Is Feature Unique?}
            J -->|Yes: Maximizes Distance from existing core| K[Add to Coreset]
            J -->|No: Highly redundant e.g., empty space| L[Discard Feature]
            K --> M[Stop at 0.1% Compression Ratio]
        end

        %% Faiss Index
        M --> N[Subsampled Coreset e.g., 5k Features]
        N --> O[Build Faiss Index]
        O --> P[(Final Memory Bank)]
    end

    %% Inference Pathway
    E -->|Inference Phase Unseen Test Image| Q[ResNet-50 Layer 1 & 2]
    
    subgraph Phase 2: Anomaly Inference & K-NN Scoring
        Q --> R[Extract Test Feature Patches]
        
        subgraph K-NN Distance Search
            R --> S[Patch 1]
            R --> T[Patch 2]
            R --> U[Patch N]
            
            S --> V{Faiss K-Nearest Neighbor}
            T --> V
            U --> V
            
            P --> V
        end
        
        %% Scoring
        V --> W[Map K-NN Distances back to Spatial Heatmap]
        W --> X[Apply Gaussian Smoothing]
        X --> Y[Find Maximum Pixel Distance]
        Y --> Z[Final Image Anomaly Score AU-ROC Evaluation]
    end
```"""

# Replace Section 1
content = re.sub(
    r"## 1\. Overall System Architecture Flow\n\nThe following graph illustrates.*?\n```\n",
    new_section_1 + "\n",
    content,
    flags=re.DOTALL
)

# 2. Remove the smaller mermaid charts in Phase 1 and Phase 2, but KEEP the text.
content = re.sub(
    r"### Phase 1: Feature Extraction & Memory Bank Construction\nDuring the training phase, the system builds a comprehensive dictionary of what \"Normal\" PRPD patches look like\.\n\n```mermaid.*?```\n\n",
    "### Phase 1: Feature Extraction & Memory Bank Construction\nDuring the training phase, the system builds a comprehensive dictionary of what \"Normal\" PRPD patches look like (as detailed in the main flowchart above).\n\n",
    content,
    flags=re.DOTALL
)

content = re.sub(
    r"### Phase 2: Anomaly Inference via K-NN Distance Search\nDuring inference, a new unseen image is passed through the same ResNet backbone, and its features are compared mathematically against the saved Memory Bank\.\n\n```mermaid.*?```\n\n",
    "### Phase 2: Anomaly Inference via K-NN Distance Search\nDuring inference, a new unseen image is passed through the same ResNet backbone, and its features are compared mathematically against the saved Memory Bank (as detailed in the main flowchart above).\n\n",
    content,
    flags=re.DOTALL
)

with open("Plans/PatchCore.md", "w") as f:
    f.write(content)
print("Updated successfully.")
