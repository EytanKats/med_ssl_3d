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

- Data preparation: for the first experiments created dataset of 1000 randomly cropped 128x128x128 patches from the NAKO dataset. Patches are cropped from volumes of water contrast of resolution 1.5x1.5x1.5, trying to have at least 70% of foreground voxels.
- The normalization of voxel values is very important for the stable training. Without normalization on a very early stage of training the NaN values appears as loss. Implemented normalization using monai.transforms.ScaleIntensityRangePercentilesd.
- Training is unstable: DINO loss for classification tokens seems to flatten after some time, while IBOT (reconstruction loss) continue to fluctuate.
  - Assumption: IBOT loss dominates training; solution: reduce IBOT loss coefficient from 1 to 0.3 but it didn't change the behavior.
 
