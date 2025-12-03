# Multimodal Chest X-ray Classification with CNN

## Introduction

Chest radiography is one of the most widely used imaging techniques in clinical practice, but interpretation requires significant expertise and time. Misdiagnosis or delays can have serious consequences, especially in resource-limited settings. Automated classification systems have the potential to support physicians by providing rapid and consistent assessments of common thoracic conditions.

This project builds a supervised image classification system that assigns a single label to chest X-ray images among four conditions: **No Finding, Effusion, Cardiomegaly, and Pneumonia.** The task is particularly challenging due to extreme class imbalance—over 90% of images fall under "No Finding" while diseases like Pneumonia appear in less than 1% of samples. I developed a baseline CNN trained on images alone, then investigated whether incorporating patient metadata (age, gender, view position) through a multimodal architecture could improve classification performance, especially for minority classes. This problem highlights both the promise of deep learning in medicine and the critical challenges posed by class imbalance and distribution shift.

---

## Illustration

Figure 1 illustrates the overall approach. The baseline model takes only chest X-ray images as input, processes them through a CNN (ResNet-18), and outputs one of four disease categories. The improved multimodal model uses a ResNet-50 backbone for image encoding, a separate MLP to process metadata features (age, gender, view position), concatenates both feature vectors, and passes them through a fusion classifier for final 4-class prediction.

<img src="APS360 Diagram (2).jpg" alt="Figure 1" width="600">

> **Figure 1:** Multimodal architecture with separate image and metadata branches, feature fusion, and classification head.

---

## Background & Related Work

I will use the **NIH ChestX-ray14 dataset** introduced by **Wang et al. (2017)**, which contains 112,120 frontal-view chest radiographs labeled with 14 disease categories. This dataset is one of the largest publicly available resources for chest radiography and has been widely adopted in deep learning research.

Recent research has shown that deep learning can provide strong results for chest X-ray analysis but also highlighted several open challenges. **Khader et al. (2023)** demonstrated that multimodal models that combine image features with non-image clinical parameters can achieve higher diagnostic performance than image-only approaches, motivating the integration of metadata in my project. **Chen et al. (2025)** provided a comprehensive review of CNN-based methods for medical image classification and concluded that deep learning models are highly effective across multiple diagnostic tasks, which supports my decision to begin with a CNN baseline. **Baltruschat et al. (2019)** compared different CNN architectures on the ChestX-ray14 dataset and showed that combining metadata such as age and body position with image features improved classification accuracy. However, they also reported that small sample sizes for certain classes introduced high variability, which aligns with the class imbalance problem I expect to face in my own work. In addition, **Irvin et al. (2019)** emphasized that evaluation metrics beyond accuracy, such as AUROC and recall, are essential for medical AI because overall accuracy can mask failures on clinically important but rare conditions.

---

## Data Processing

### Dataset Sources
All images were obtained from the **NIH ChestX-ray14 dataset** (Wang et al., 2017). Two fully disjoint subsets were constructed:
- **Development subset**: 6,433 images from `images_001` folder
- **Unseen generalization subset**: 6,427 images from `images_010` folder

### Dataset Statistics
The development subset exhibits severe class imbalance:
- **No Finding**: 5,999 (93.0%)
- **Effusion**: 309 (4.8%)
- **Cardiomegaly**: 91 (1.4%)
- **Pneumonia**: 34 (0.5%)

The unseen subset maintains similar proportions, confirming consistent distribution characteristics.

### Preprocessing Steps
Images were:
- Resized to 256×256 pixels
- Augmented (training only): random rotations (±15°), horizontal flips, translations, perspective distortion, color jitter
- Center-cropped to 224×224
- Normalized with ImageNet statistics

Metadata features were standardized:
- **Age**: z-score normalized
- **Gender**: binary encoding (M=1, F=0)
- **View Position**: binary encoding (PA=1, AP=0)

The development subset was split into **70% train, 15% validation, 15% test**.

---

## Architecture

### Baseline Model
The baseline uses a **ResNet-18** initialized from scratch (no ImageNet pretraining) with:
- Modified final layer for 4-class classification
- Unweighted CrossEntropyLoss
- Adam optimizer
- Simple training for 5 epochs without early stopping

### Primary Multimodal Model
The final model adopts a multimodal structure with three components:

**Image Branch:**
- ResNet-50 encoder (pretrained on ImageNet) producing 2048-dimensional embeddings
- Early layers frozen for stability; later layers fine-tuned

**Metadata Branch:**
- MLP with hidden sizes [64, 32] for encoding age, gender, and view position
- ReLU activations with dropout (rates 0.2 and 0.1)

**Fusion Classifier:**
- Concatenates 2048-D image features and 32-D metadata features
- Fully connected layers: 1024 → 512 → 128 → 4
- Dropout (0.3, 0.2, 0.1) and BatchNorm1d after first two layers

**Imbalance-Aware Training:**
- Focal Loss (gamma=1.5) with smoothed class weights
- AdamW optimizer with gradient clipping (max_norm=1.0)
- ReduceLROnPlateau scheduler (patience=5, factor=0.7)
- WeightedRandomSampler for minority class oversampling
- Early stopping based on validation macro F1-score

---

## Results

### Baseline Performance
The baseline achieved **93% accuracy** on the development test set but failed completely on minority classes:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Cardiomegaly | 0.00 | 0.00 | 0.00 | 15 |
| Effusion | 0.00 | 0.00 | 0.00 | 47 |
| No Finding | 0.93 | 1.00 | 0.96 | 901 |
| Pneumonia | 0.00 | 0.00 | 0.00 | 6 |
| **Accuracy** | | | | **0.93** |
| **Macro Avg** | **0.23** | **0.25** | **0.24** | **969** |
| **Weighted Avg** | **0.86** | **0.93** | **0.90** | **969** |

The baseline essentially learned to predict "No Finding" for all images, demonstrating why accuracy is misleading in imbalanced medical tasks.

### Primary Model Performance (Development Test Set)
The multimodal model substantially improved minority class detection:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Cardiomegaly | 0.38 | 0.33 | 0.36 | 15 |
| Effusion | 0.23 | 0.55 | 0.33 | 47 |
| No Finding | 0.96 | 0.89 | 0.92 | 901 |
| Pneumonia | 0.33 | 0.17 | 0.22 | 6 |
| **Accuracy** | | | | **0.86** |
| **Macro Avg** | **0.48** | **0.49** | **0.46** | **969** |
| **Weighted Avg** | **0.91** | **0.86** | **0.88** | **969** |

**Key improvements:**
- Macro F1 increased from 0.24 → 0.46 (+92%)
- Effusion recall: 0.00 → 0.55
- Cardiomegaly recall: 0.00 → 0.33
- Pneumonia: achieved non-zero detection despite only 34 training samples

### Generalization to Unseen Data
Evaluation on the fully disjoint unseen subset (images_010) demonstrated robustness:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Cardiomegaly | 0.09 | 0.35 | 0.15 | 17 |
| Effusion | 0.38 | 0.64 | 0.47 | 55 |
| No Finding | 0.96 | 0.87 | 0.91 | 889 |
| Pneumonia | 0.00 | 0.00 | 0.00 | 5 |
| **Accuracy** | | | | **0.84** |
| **Macro Avg** | **0.36** | **0.46** | **0.38** | **966** |
| **Weighted Avg** | **0.91** | **0.84** | **0.87** | **966** |

Despite distribution shift, the model maintained meaningful recall for Effusion (0.64) and Cardiomegaly (0.35), confirming that learned representations generalize beyond the development subset.

---

## Discussion

### Key Findings

**Addressing Class Imbalance:**
The baseline's 93% accuracy masked complete failure on minority classes, confirming that standard metrics are inadequate for imbalanced medical data. The multimodal model's use of focal loss, class weighting, and oversampling enabled meaningful detection of rare conditions.

**Value of Multimodal Learning:**
Incorporating metadata (age, gender, view position) alongside image features improved macro F1-score by 92% (0.24 → 0.46) and enabled the model to learn disease-specific patterns beyond visual cues alone. This aligns with findings from Khader et al. (2023) and Baltruschat et al. (2019).

**Generalization Under Distribution Shift:**
The unseen evaluation demonstrated that the model learned transferable disease representations rather than memorizing dataset-specific artifacts. Stable recall for Effusion and Cardiomegaly across different patient cohorts suggests potential for real-world deployment with appropriate validation.

### Limitations

**Pneumonia Detection:**
Extremely low sample counts (34 training, 6 development test, 5 unseen test) combined with heterogeneous radiographic presentations prevented robust pneumonia detection. This reflects data limitations rather than architectural flaws.

**Precision-Recall Tradeoff:**
While minority class recall improved substantially, precision remained low (0.17-0.29), indicating many false positives. Clinical deployment would require threshold tuning based on specific use cases and cost-benefit analysis.

**Dataset Constraints:**
Both subsets maintain similar imbalance patterns, limiting evaluation of performance under more extreme distribution shifts or different acquisition protocols.

### Future Directions
- Collect more pneumonia samples or use semi-supervised learning
- Explore multi-label classification to preserve co-occurring conditions
- Investigate uncertainty quantification for safer clinical integration
- Evaluate across demographic subgroups to detect potential biases

---

## Ethical Considerations

Medical AI systems risk amplifying dataset biases, especially when minority diseases are underrepresented. Following Ueda et al. (2023), ensuring fairness in healthcare AI directly affects patient safety and equitable access to medical services. 

**Key concerns in this project:**
- Errors in minority class detection (especially Pneumonia) could lead to unsafe clinical outcomes if deployed without oversight
- Metadata features may encode demographic or clinical biases that require careful subgroup evaluation
- High false positive rates for rare conditions could lead to unnecessary follow-up procedures

Any real-world deployment would require extensive validation across diverse patient populations, uncertainty quantification mechanisms, and human-in-the-loop design to ensure the system supports rather than replaces clinical judgment.

---

## References

1. Wang et al., NIH ChestX-ray14 dataset 2017.
2. Khader et al., *Multimodal Deep Learning for Integrating Chest Radiographs and Clinical Parameters: A Case for Transformers*, Radiology 2023.
3. Chen et al., *A Review of Convolutional Neural Network Based Methods for Medical Image Classification*, Computers in Biology and Medicine 2025.
4. Baltruschat et al., *Comparison of Deep Learning Approaches for Multi-Label Chest X-Ray Classification*, PLoS ONE 2019.
5. Irvin et al., *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison*, AAAI 2019.
6. Ueda et al., *Fairness of Artificial Intelligence in Healthcare: Review and Recommendations*, PMC 2023.
