# FaCER: Facial Counterfactual Generation via Causal Mask-Guided Editing

**Published in Transactions on Machine Learning Research (TMLR), March 2026**

[[Paper]](https://openreview.net/forum?id=ssamEGQj0C)

## Overview

FaCER is a neuro-symbolic framework for generating causally consistent counterfactual facial images. It integrates three key components:

1. **Causal Graph Discovery** -- Uses the Fast Causal Inference (FCI) algorithm to uncover latent causal relationships among facial attributes, producing a Partial Ancestral Graph (PAG) that distinguishes direct causes from confounders.
2. **Mask-Guided Counterfactual Generation** -- Constructs spatially informed masks from the learned causal graph to guide a DDPM-based diffusion model, ensuring only causally relevant facial regions are modified.
3. **Semantic Interpretation** -- Leverages CLIP-based embeddings to provide human-understandable textual explanations of the visual changes in generated counterfactuals.

Unlike existing methods that treat facial attributes as independently manipulable, FaCER respects causal dependencies between attributes, producing counterfactuals that are realistic, sparse, and free from stereotype bias.

## Pipeline

```
Input Image
    |
    v
Causal Graph Learning (FCI on facial attribute matrix)
    |
    v
Reference Image Generation (DDPM with classifier guidance)
    |
    v
Preliminary Image Generation (DDPM with weak guidance)
    |
    v
Causal Mask Construction (pixel diff filtered by direct causes from PAG)
    |
    v
Mask-Guided Counterfactual Generation (inpaint only causal regions)
    |
    v
CLIP-based Semantic Interpretation (textual explanation of edits)
```

## Project Structure

```
FaCER/
├── main.py                        # Main entry point for counterfactual generation
├── postprocessing.py              # Post-processing of generated counterfactuals
├── counterfactual_evaluation.py   # CLIP-based counterfactual evaluation
├── facer_clip.py                  # CLIP-based captioning and interpretation
├── calculate_fairness.py          # Fairness metrics (DI, SPD)
├── batch_clip_similarity.py       # Batch CLIP similarity computation
├── find_fci.py                    # Fast Causal Inference graph discovery
│
├── core/
│   ├── utils.py                   # Mask generation, facial landmarks
│   ├── metrics.py                 # Accuracy and prediction utilities
│   ├── attacks_and_models.py      # Adversarial attack implementations (PGD, C&W, GD)
│   ├── DCC_flow.py                # Diffusion-based CF generation flow
│   └── pyramid_flow.py            # Pyramid-based flow
│
├── guided_diffusion/              # Diffusion model implementation
│   ├── gaussian_diffusion.py      # DDPM/DDIM diffusion logic
│   ├── unet.py                    # UNet denoising architecture
│   ├── CFModel.py                 # Counterfactual model
│   ├── sample_utils.py            # Sampling and dataset utilities
│   ├── script_util.py             # Model/diffusion factory functions
│   ├── image_datasets.py          # Dataset loaders (CelebA, CelebA-HQ, etc.)
│   ├── train_util.py              # Training utilities
│   └── ...                        # Additional modules (losses, FP16, logging, etc.)
│
├── eval_utils/                    # Evaluation utilities
│   ├── fid_metrics.py             # FID score computation
│   ├── cout_metrics.py            # Confusion-based metrics
│   ├── oracle_celeba_metrics.py   # CelebA oracle metrics
│   └── oracle_celebahq_metrics.py # CelebA-HQ oracle metrics
│
├── scripts/                       # Experiment launch scripts
│   ├── celeba-attack.sh           # CelebA smile attribute
│   ├── celeba-attack_y.sh         # CelebA age attribute
│   ├── celeba-attack-g.sh         # CelebA gender attribute
│   ├── celebahq-attack.sh         # CelebA-HQ experiments
│   └── ...                        # Additional experiment configs
│
└── TMLR_FaCER_CameraReady.pdf     # Published paper
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (GPU required)

### Dependencies

```bash
pip install torch torchvision numpy scipy pillow opencv-python dlib
pip install pandas scikit-learn matplotlib tqdm pyyaml
pip install git+https://github.com/openai/CLIP.git
pip install causal-learn   # for FCI causal discovery

# Optional: for distributed training
pip install mpi4py blobfile
```

### Pre-trained Models

Pre-trained DDPM models and classifiers can be obtained from the [ACE (Adversarial Counterfactual Visual Explanations)](https://github.com/guillaumejs2403/ACE) repository. Follow their instructions to download:

| Model | Description | Source |
|---|---|---|
| `ddpm-celeba.pt` | DDPM trained on CelebA (128x128) | [ACE repo](https://github.com/guillaumejs2403/ACE) |
| `celebahq-ddpm.pt` | DDPM trained on CelebA-HQ (256x256) | [openai/guided-diffusion](https://github.com/openai/guided-diffusion) |
| `classifier.pth` | Facial attribute classifier for CelebA | [ACE repo](https://github.com/guillaumejs2403/ACE) |
| `checkpoint.tar` | DenseNet classifier for CelebA-HQ | [ACE repo](https://github.com/guillaumejs2403/ACE) |

## Usage

### Counterfactual Generation

Generate counterfactual explanations using the main pipeline:

```bash
# CelebA -- Smile attribute (Q=31)
python main.py \
    --attention_resolutions 32,16,8 --class_cond False \
    --diffusion_steps 500 --learn_sigma True --noise_schedule linear \
    --num_channels 128 --num_heads 4 --num_res_blocks 2 \
    --resblock_updown True --use_fp16 False --use_scale_shift_norm True \
    --batch_size 1 --gpu 0 \
    --num_samples 5000 \
    --model_path ./ddpm-celeba.pt \
    --classifier_path ./classifier.pth \
    --output_path ./celeba_results \
    --exp_name CelebA \
    --attack_method PGD \
    --attack_iterations 50 \
    --attack_joint True \
    --dist_l1 0.001 \
    --timestep_respacing 50 \
    --sampling_time_fraction 0.1 \
    --sampling_stochastic True \
    --sampling_inpaint 0.15 \
    --label_query 31 \
    --label_target -1 \
    --image_size 128 \
    --data_dir ./img_align_celeba \
    --dataset CelebA
```

Or use the provided scripts:

```bash
bash scripts/celeba-attack.sh    # CelebA smile
bash scripts/celeba-attack_y.sh  # CelebA age
bash scripts/celebahq-attack.sh  # CelebA-HQ smile
```

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--label_query` | Target attribute index (31=Smile, 39=Age, 20=Gender) | -- |
| `--label_target` | Target label (-1 for flip) | -1 |
| `--attack_method` | Attack type: `PGD`, `GD`, `CW`, or `None` | PGD |
| `--attack_iterations` | Number of adversarial attack iterations | 50 |
| `--sampling_time_fraction` | Fraction of diffusion timesteps to use | 0.1 |
| `--sampling_inpaint` | Inpainting threshold for mask-guided generation | 0.15 |
| `--dataset` | Dataset: `CelebA`, `CelebAHQ` | -- |
| `--image_size` | Image resolution (128 for CelebA, 256 for CelebA-HQ) | 128 |

### Evaluation

```bash
# CLIP-based semantic evaluation
python counterfactual_evaluation.py

# Fairness metrics (Disparate Impact, Statistical Parity Difference)
python calculate_fairness.py

# Batch CLIP similarity
python batch_clip_similarity.py
```

### Causal Graph Discovery

```bash
# Run FCI algorithm on facial attribute data
python find_fci.py
```

## Datasets

| Dataset | Images | Resolution | Attributes |
|---|---|---|---|
| [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | 200K+ | 178x218 | 40 binary attributes |
| [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ) | 30K | 1024x1024 | 40 binary attributes |

## Evaluation Metrics

- **FID / sFID** -- Realism of generated counterfactuals (lower is better)
- **MNAC** -- Mean Number of Attributes Changed; measures sparsity (lower is better)
- **FVA** -- Face Verification Accuracy; measures identity preservation (higher is better)
- **DI** -- Disparate Impact; measures group fairness (closer to 1 is better)
- **SPD** -- Statistical Parity Difference; measures prediction parity (closer to 0 is better)
- **R@1** -- CLIP Recall@1; measures semantic alignment of explanations

## Citation

```bibtex
@article{tan2026facial,
  title={Facial Counterfactual Generation via Causal Mask-Guided Editing},
  author={Tan, Pei Sze and Rajanala, Sailaja and Pal, Arghya and Phan, Rapha{\"e}l C.-W. and Ong, Huey-Fang},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
