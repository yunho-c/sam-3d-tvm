# SAM-3D Objects: Theory of Operation

> **SAM 3D Objects** is a foundation model from Meta that reconstructs full 3D shape geometry, texture, and layout from a single image. It excels in real-world scenarios with occlusion and clutter using progressive training and a human-feedback-enhanced data engine.

---

## System Architecture Overview

```mermaid
flowchart TB
    subgraph Input ["**Input Processing**"]
        IMG[/"RGB Image"/]
        MASK[/"Segmentation Mask"/]
        MERGE["Merge to RGBA"]
    end
    
    subgraph Conditioning ["**Image Conditioning**"]
        DINO["DINOv2 ViT-B/14<br/>Image Encoder"]
        COND_EMB["Condition Embedder<br/>(256 tokens)"]
    end
    
    subgraph Stage1 ["**Stage 1: Sparse Structure Generation**"]
        SS_GEN["Sparse Structure<br/>Flow Matching Generator<br/>(DiT-based)"]
        SS_DEC["Sparse Structure<br/>VAE Decoder"]
        VOXEL[("64³ Binary<br/>Voxel Grid")]
    end
    
    subgraph Stage2 ["**Stage 2: Structured Latent Generation**"]
        SLAT_GEN["Sparse Latent (SLAT)<br/>Flow Matching Generator<br/>(Sparse DiT)"]
        SPARSE_TENSOR[("SparseTensor<br/>Features")]
    end
    
    subgraph Output ["**Output Decoders**"]
        GS_DEC["Gaussian Splatting<br/>Decoder"]
        MESH_DEC["Mesh Decoder"]
        GS_OUT[/"3D Gaussians<br/>(PLY)"/]
        MESH_OUT[/"Textured Mesh<br/>(GLB)"/]
    end
    
    IMG --> MERGE
    MASK --> MERGE
    MERGE --> DINO
    DINO --> COND_EMB
    
    COND_EMB --> SS_GEN
    SS_GEN -->|"Flow Matching<br/>25 steps"| SS_DEC
    SS_DEC --> VOXEL
    
    VOXEL -->|"coords"| SLAT_GEN
    COND_EMB --> SLAT_GEN
    SLAT_GEN -->|"Flow Matching<br/>25 steps"| SPARSE_TENSOR
    
    SPARSE_TENSOR --> GS_DEC
    SPARSE_TENSOR --> MESH_DEC
    GS_DEC --> GS_OUT
    MESH_DEC --> MESH_OUT
```

---

## Core Pipeline Stages

### Stage 1: Sparse Structure Generation

The first stage predicts a **coarse 3D occupancy grid** from the input image:

```mermaid
flowchart LR
    subgraph SS_Pipeline ["Sparse Structure Pipeline"]
        direction LR
        A["Latent Noise<br/>(B, 4096, 8)"] 
        B["SS Flow Model<br/>(DiT Transformer)"]
        C["SS VAE Decoder<br/>(3D CNN)"]
        D["Binary Voxel<br/>(64³)"]
        
        A -->|"Denoising<br/>(25 steps)"| B
        B -->|"Reshape to<br/>(B, 8, 16, 16, 16)"| C
        C -->|"Threshold > 0"| D
    end
```

**Key characteristics:**
- Generates a **16³ latent cube** with 8 channels, then upsamples to **64³ voxels**
- Uses Classifier-Free Guidance (CFG) with strength=7
- Flow matching with rectified flow for stable generation
- Outputs coordinates of occupied voxels for Stage 2

### Stage 2: Structured Latent (SLAT) Generation

The second stage generates **detailed 3D features** at occupied voxel locations:

```mermaid
flowchart LR
    subgraph SLAT_Pipeline ["Structured Latent Pipeline"]
        direction LR
        A["Sparse Noise<br/>(N_voxels, 8)"]
        B["SLAT Flow Model<br/>(Sparse Transformer)"]
        C["SparseTensor<br/>Features"]
        
        A -->|"Sparse Denoising<br/>(25 steps)"| B
        B --> C
    end
```

**Key characteristics:**
- Operates only on **occupied voxel coordinates** from Stage 1
- Uses `SparseTensor` representation for memory efficiency
- Sparse attention with coordinate-aware positional embeddings
- CFG with strength=5 for balanced quality/diversity

---

## Key Components

### Model Architecture Components

| Component | Class | Purpose | Key Features |
|-----------|-------|---------|--------------|
| **SS Generator** | `SparseStructureFlowModel` | Generates 3D structure latent | DiT-style transformer, cross-attention to image tokens |
| **SLAT Generator** | `SLatFlowModel` | Generates sparse 3D features | Sparse transformer with SparseResBlock3d |
| **SS Decoder** | `SparseStructureDecoder` | Decodes latent to voxels | 3D CNN with upsampling blocks |
| **GS Decoder** | (in `representations/gaussian`) | Decodes to Gaussians | Predicts xyz, rotation, scaling, opacity, SH colors |
| **Mesh Decoder** | (in `representations/mesh`) | Decodes to mesh | Predicts vertices and faces |
| **Condition Embedder** | `Dino` | Encodes input image | DINOv2 ViT-B/14, 256 output tokens |

### Flow Matching with CFG

| Parameter | SS Generator | SLAT Generator |
|-----------|-------------|----------------|
| **Inference Steps** | 25 | 25 |
| **CFG Strength** | 7.0 | 5.0 |
| **CFG Interval** | [0, 500] | [0, 500] |
| **Rescale t** | 3 | 3 |

---

## Sparse Tensor Representation

SAM-3D uses a custom `SparseTensor` class that supports both **torchsparse** and **spconv** backends:

```mermaid
classDiagram
    class SparseTensor {
        +Tensor feats
        +Tensor coords [B, X, Y, Z]
        +Size shape
        +List~slice~ layout
        +to(device, dtype)
        +replace(feats, coords)
        +dense() Tensor
        +unbind(dim) List
    }
    
    class sparse_ops {
        +sparse_cat(inputs, dim)
        +sparse_unbind(input, dim)
        +sparse_batch_broadcast()
        +sparse_batch_op()
    }
    
    SparseTensor --> sparse_ops : uses
```

**Features:**
- **Batch dimension in coords[:, 0]** - contiguous data per batch
- **Layout slices** track per-batch data ranges
- **Spatial cache** for efficient repeated operations
- Supports element-wise operations (+, -, *, /)

---

## Gaussian Splatting Output

The `Gaussian` class represents 3D Gaussian splats:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `xyz` | (N, 3) | 3D positions |
| `scaling` | (N, 3) | Gaussian scales (log-space) |
| `rotation` | (N, 4) | Quaternion rotations |
| `opacity` | (N, 1) | Opacity values (sigmoid-inverse) |
| `features_dc` | (N, 3) | Base RGB (SH degree 0) |
| `features_rest` | (N, 45) | Higher SH coefficients |

---

## Inference Pipeline Flow

```mermaid
sequenceDiagram
    participant User
    participant Inference
    participant SS_Gen as SS Generator
    participant SS_Dec as SS Decoder
    participant SLAT_Gen as SLAT Generator
    participant Decoders as GS/Mesh Decoders
    
    User->>Inference: image, mask, seed
    Inference->>Inference: merge_image_and_mask()
    Inference->>Inference: preprocess_image()
    
    rect rgb(240, 240, 255)
        Note over SS_Gen,SS_Dec: Stage 1: Sparse Structure
        Inference->>SS_Gen: sample_sparse_structure()
        SS_Gen->>SS_Gen: embed_condition(DINOv2)
        SS_Gen->>SS_Gen: flow_matching(25 steps, CFG=7)
        SS_Gen->>SS_Dec: latent → voxels
        SS_Dec-->>Inference: coords (occupied voxels)
    end
    
    rect rgb(255, 240, 240)
        Note over SLAT_Gen: Stage 2: Structured Latent
        Inference->>SLAT_Gen: sample_slat(coords)
        SLAT_Gen->>SLAT_Gen: embed_condition(DINOv2)
        SLAT_Gen->>SLAT_Gen: flow_matching(25 steps, CFG=5)
        SLAT_Gen-->>Inference: SparseTensor(slat)
    end
    
    rect rgb(240, 255, 240)
        Note over Decoders: Output Decoding
        Inference->>Decoders: decode_slat(slat)
        Decoders-->>Inference: gaussian, mesh
        Inference->>Inference: postprocess (texture baking)
    end
    
    Inference-->>User: {gs, mesh, glb, ...}
```

---

## Directory Structure

```
sam3d_objects/
├── model/
│   ├── backbone/
│   │   ├── dit/
│   │   │   └── embedder/
│   │   │       ├── dino.py          # DINOv2 image encoder
│   │   │       └── pointmap.py       # Optional point cloud conditioning
│   │   ├── generator/
│   │   │   ├── classifier_free_guidance.py  # CFG wrapper
│   │   │   └── flow_matching/        # Flow matching samplers
│   │   └── tdfy_dit/
│   │       ├── models/
│   │       │   ├── sparse_structure_flow.py    # Stage 1 generator
│   │       │   ├── structured_latent_flow.py   # Stage 2 generator
│   │       │   └── sparse_structure_vae.py     # VAE encoder/decoder
│   │       ├── modules/
│   │       │   ├── sparse/           # SparseTensor, sparse conv/attention
│   │       │   └── transformer/      # Transformer blocks
│   │       └── representations/
│   │           ├── gaussian/         # 3D Gaussian output
│   │           └── mesh/             # Mesh output
│   └── io.py                         # Model loading utilities
├── pipeline/
│   ├── inference_pipeline.py         # Main inference orchestration
│   ├── inference_utils.py            # Helper functions
│   └── layout_post_optimization_utils.py  # Scene optimization
├── data/
│   └── dataset/                      # Data loading
└── utils/
    └── visualization.py              # Scene visualization
```

---

## Key Technical Insights

### 1. Two-Stage Generation
- **Stage 1** focuses on coarse geometry (where is the object?)
- **Stage 2** focuses on detailed appearance (what does it look like?)
- This decomposition enables handling of complex occlusions

### 2. Sparse Representations
- Only process occupied voxels → memory efficient for high resolution
- SparseTensor abstracts torchsparse/spconv backends
- Enables scaling to 64³ and beyond

### 3. Flow Matching
- Uses rectified flows for generation (not DDPM/DDIM)
- `rescale_t=3` concentrates steps near t=0 (denoising end)
- CFG applied with strength interpolation over timesteps

### 4. Multi-Modal Outputs
- Same latent can decode to **Gaussian Splats** (for rendering) or **Mesh** (for export)
- Texture baking projects Gaussian colors onto mesh UVs
- GLB export with simplification (95% triangle reduction)

### 5. torch.compile Support
- `_compile()` method enables max-autotune compilation
- Wraps condition embedder, SS generator, and SS decoder
- Warmup runs for kernel specialization

---

## Model Size & Resources

| Model Component | Approximate Parameters |
|-----------------|----------------------|
| DINOv2 ViT-B/14 | ~86M |
| SS Generator (DiT) | ~300M |
| SLAT Generator (Sparse DiT) | ~400M |
| SS Decoder (3D CNN) | ~20M |
| SLAT Decoder GS | ~50M |
| SLAT Decoder Mesh | ~50M |

**Requirements:**
- GPU with 16GB+ VRAM recommended
- CUDA support required
- Flash Attention enabled for A100/H100/H200

---

## References

- **Paper**: [SAM 3D: 3Dfy Anything in Images](https://arxiv.org/abs/2511.16624)
- **Code**: [github.com/facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects)
- **Related**: [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) for human mesh recovery
