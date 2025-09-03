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

## 👨‍💻 Development

## Data Preparation
- For the first experiments, created a dataset of **1000 randomly cropped 3D patches** of size **128×128×128** from the **NAKO dataset**.  
- Patches are cropped from **water contrast volumes** with isotropic resolution **1.5×1.5×1.5 mm³**, ensuring at least **70% of voxels are foreground**.  

## Normalization
- **Normalization of voxel values** is very important for stable training.  
- Without normalization, **NaN values** appeared in the loss function at a very early stage.  
- Implemented normalization using:  
  - [`monai.transforms.ScaleIntensityRangePercentilesd`](https://docs.monai.io/en/stable/transforms.html#scaleintensityrangepercentilesd).  

## Training Instability
- **Observation:**  
  - DINO loss (classification tokens) seems to **flatten** after some time.  
  - IBOT loss (reconstruction) continues to **fluctuate**.  

- **Hypotheses & Attempts:**  
  1. **IBOT loss dominates training**  
     - Attempt: reduce IBOT loss coefficient from **1 → 0.3**.  
     - Result: no improvement, DINO loss still flat.  

  2. **DINO loss applied on small batch & global views only**  
     - Concern:  
       - Very small batch of images.  
       - Global views cropped and resized, not very different, so pushing them apart may be confusing.  
     - Attempt: restrict cropping scale range to **0.8–1.0**.  
     - Result: DINO loss no longer flat but **fluctuates without decreasing**.  

  3. **Augmentation interference**  
     - Attempt: disable augmentations (`Resized scaling`, `Affine`, `Histogram shift`, `Gaussian smoothing`).  
     - Kept only **masking between teacher and student**.  
     - Result: DINO loss lower but again fluctuates without further decrease.  
    
  4. **Freeze last layer and adjust base learning rate**  
   - Attempt:  
     1. Froze last layer of DINO & iBOT heads for 30 epochs (not applied previously) to prevent collapse into uniform embeddings.  
     2. Increased base learning rate from **0.0002 → 0.01** to account for very small batch sizes where DINO’s scaling rule may not apply.  
   - Result: After unfreezing the last layer, both DINO and iBOT losses decreased in a more stable manner. However, iBOT loss still shows only limited decrease compared to expectations.
