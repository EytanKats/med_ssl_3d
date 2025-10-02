# 🧠 Medical Self-Supervised Learning

## 🚀 Research Directions
- **Multimodal CT-MRI pre-training** using Multi-MAE approach  
- **Bilateral dense contrastive learning**  
- **Registration as a pretext task** for self-supervised pre-training  
- **CroCo-based self-supervised pre-training** for improving registration downstream tasks 

---

## 📚 Resources

### 🔗 Repositories

| Paper | Repository | Stage | Method/Task | Architecture | Description |
|-------|------------|-------|-------------|--------------|-------------|
| NA | [**3D DINO (AIM)**](https://github.com/AIM-Harvard/DINOv2-3D-Med) | Pre-training | Self-supervised representation learning | Transformer (ViT-based, DINOv2) | 3D implementation of DINOv2 |
| NA | [**3D DINO (AICONS)**](https://github.com/AICONSlab/3DINO) | Pre-training | Self-supervised representation learning | Transformer (ViT-based, DINOv2) | 3D implementation of DINOv2 |
| NA | [**Foundation-based Registration**](https://github.com/mazurowski-lab/Foundation-based-reg) | Zero-shot / Training-free | Medical image registration | Vision foundation models (encoder backbone) | Training-free (zero-shot) medical image registration pipeline using vision foundation models as feature encoders |
| NA | [**nnssl Pre-training**](https://github.com/MIC-DKFZ/nnssl?tab=readme-ov-file#complimentary-resources) | Pre-training | Self-supervised learning | CNN & Transformer (nnU-Net framework) | 3D implementation of strong pre-training methods using both CNN and Transformer architectures based on nnU-Net framework |
| NA | [**nnssl Fine-tuning**](https://github.com/TaWald/nnUNet) | Fine-tuning | Downstream segmentation | CNN (nnU-Net) | Downstream segmentation pretraining and adaptation framework based on nnU-Net framework |

---

## 👨‍💻 Development Journal

### Data Preparation
- For the first experiments, created a dataset of **1000 randomly cropped 3D patches** of size **128×128×128** from the **NAKO dataset**.  
- Patches are cropped from **water contrast volumes** with isotropic resolution **1.5×1.5×1.5 mm³**, ensuring at least **70% of voxels are foreground**.  

### Normalization
- **Normalization of voxel values** is very important for stable training.  
- Without normalization, **NaN values** appeared in the loss function at a very early stage.  
- Implemented normalization using:  
  - [`monai.transforms.ScaleIntensityRangePercentilesd`](https://docs.monai.io/en/stable/transforms.html#scaleintensityrangepercentilesd).  

### ⚠️ DINOv2 Training Instability Debugging

#### 🔍 Observations
- DINO loss (classification tokens) tends to **flatten** after some time.  
- iBOT loss (reconstruction) continues to **fluctuate**.  

#### 🧪 Hypotheses & Experiments

1. **IBOT loss dominates training**  
   - **Attempt:** Reduced IBOT loss coefficient from **1 → 0.3**.  
   - **Result:** No improvement, DINO loss remained flat.  

2. **DINO loss with small batch & global views only**  
   - **Concern:**  
     - Very small batch size of images.  
     - Global views cropped/resized are not sufficiently different, making it confusing when pushing them apart.  
   - **Attempt:** Restricted cropping scale range to **0.8–1.0**.  
   - **Result:** DINO loss no longer flat, but **fluctuates without decreasing**.  

3. **Augmentation interference**  
   - **Attempt:** Disabled augmentations (`Resized scaling`, `Affine`, `Histogram shift`, `Gaussian smoothing`), keeping only **masking between teacher and student**.  
   - **Result:** DINO loss became lower, but continued to fluctuate without further decrease.  

4. **Freeze last layer & adjust base learning rate**  
   - **Attempt:**  
     1. Kept augmentations disabled.  
     2. Froze last layer of DINO & iBOT heads for **4500 steps** (not applied before) to prevent collapse into uniform embeddings.  
     3. Increased base LR from **0.0002 → 0.01** to better handle very small batch sizes (DINO scaling rule not reliable here).  
   - **Result:** After unfreezing, both DINO and iBOT losses decreased more stably. However, iBOT loss still showed only **limited decrease** compared to expectations.  

5. **Shorter freeze, restore augmentations, adjust temperature scaling**  
   - **Attempt:**  
     1. Kept LR at **0.01** (helpful for convergence).  
     2. Shortened last-layer freeze to **1500 steps** (longer freeze suspected to limit iBOT convergence).  
     3. Reintroduced augmentations to make DINO task harder.  
     4. Extended teacher temperature warm-up from **1500 → 4500 steps** to stabilize early training.  
   - **Result:** DINO loss plateaued; iBOT loss decreased initially but then **increased again** after some time.  

6. **Cropping size of global views**  
   - **Attempt:** Restricted global cropping scale range to **0.8–1.0** to reduce confusion.  
   - **Result:** Both DINO and iBOT losses decreased initially but later **increased again**. Fluctuations at the beginning persisted, even after unfreezing heads.  

7. **Increase patch size**  
   - **Attempt:** Increased patch size from **8 → 16**.  
     - Small patches in medical images may be too limited in context, making embeddings unstable.  
     - Larger patches reduce number of tokens, provide more context, and lower memory consumption.  
   - **Result:** iBOT loss became lower (expected: fewer patches, more context, better embeddings). However, the same issue persisted—losses eventually **increased again**, and fluctuations remained.  

8. **Increase batch size & train longer**  
   - **Attempt:**  
     - Increased batch size (may reduce fluctuations).  
     - Extended training duration (loss instability might stem from teacher temperature reaching its max value too early).  
   - **Result:** Both DINO and iBOT losses decreased lower than before, but later in training **both increased again**. iBOT loss fluctuations remained.  

9. **Apply DINO loss on unmasked input**  
   - **Attempt:**  
     - Forward pass student twice:  
       - On **unmasked input** → compute DINO loss.  
       - On **masked input** → compute iBOT loss.  
   - **Result:** DINO loss converged well. iBOT loss decreased more than before but still **increased again later**, with persistent fluctuations.  

10. **Teacher update too slow at later stages**  
    - **Hypothesis:**  
      - Teacher momentum increased too quickly.  
      - This caused slow teacher updates while the teacher was still weak, leading to growing discrepancies between teacher and student models and divergence of iBOT loss.  
    - **Attempt:** Extended training for more epochs.  
    - **Result:** iBOT convergence improved significantly, but **fluctuations remained**.  

11. **Training instability (NaN losses)**  
   - **Observation:**  
     - Training occasionally diverges with **NaN losses**, although some runs still reach convergence.  
   - **Hypothesis:**  
     - A **base learning rate of 0.01** may be too aggressive for the **AdamW optimizer**, amplifying numerical instabilities and causing divergence.  
   - **Attempt:**  
     - Lowered base learning rate from **0.01 → 0.001** to improve stability.  
   - **Result:**  
     - Both **DINO** and **iBOT** losses now converge more steadily, though overall convergence is slower.  
     - After roughly **170 k steps**, the iBOT loss begins to **increase again**.  
     - This late-stage rise may stem from the **teacher–student gap widening** as teacher updates slow (high momentum) combined with **limited data**, leaving the teacher insufficiently strong to guide the student effectively.


### 🦖 Using the DINOReg Approach for Zero-Shot Registration

- **Overview of the DINOReg approach**:
  - The outputs of the pre-trained model for both moving and fixed images are concatenated.  
  - A dimensionality reduction method (e.g., PCA) is applied to reduce the number of features (e.g., to 12).  
  - The features of the moving and fixed images are reshaped and resized to match the original image dimensions.  
  - The **ConvexAdam** algorithm is applied to the resulting feature maps to obtain the deformation field.  

- **Key considerations**:
  - The accuracy of the deformation is highly dependent on the resolution of the feature maps produced by the model.  
  - When the output resolution is low, precise image alignment is difficult. For instance, patch sizes of 14 or 16 often result in suboptimal performance.  

- **Limitations and workaround**:
  - A common solution (used in the original DINOReg) is to upsample the image before feeding it into the model.  
  - However, this approach significantly increases memory consumption and does not scale well.  

- **Suggested improvement**:
  - Employ the **SegFormer architecture**, which computes attention across multiscale volumetric features.  
  - It uses a lightweight all-MLP decoder to efficiently aggregate local and global attention, enabling the generation of effective dense feature representations. 
