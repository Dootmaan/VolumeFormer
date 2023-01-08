# Essential Experiment Code for VolumeFormer

**Please note that the current code is only intended for verifying the experimental results in our paper. We are sorry that due to the limited time many part of it is still very raw. The final version of our open code will be more complete and easy to use**

- The code includes ablation study and comparisons with other patch-based or patch-free vision transformers. Some results are borrowed from [PFSeg](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_13). The experimental settings is also exactly the same as those of PFSeg. 

- **Remember to modify config/config.py for different input size.** For example, the default config is for patch-free methods such as UNETR and VolumeFormer, while the second config is for SwinUNETR.

- It you would like to verify the results of AD+VTB (w/o weight sharing), AD+VTB (w/o MSRP) and AD+VTB (w/ linear V mapping) please modify the code in model/VolumeFormer.py manually.

---

## Quick introduction about Adaptive Decomposition and Shared Weight Volumetric Transformer Block

- Adaptive Decomposition is the generalized decomposition method which can realize "GPU memory-efficient" patch-free segmentation. The previously proposed Holistic Decomposition is a special case of it. 

- Shared Weight Volumetric Transformer Block is a highly efficient long-range dependencies modeling module. We first propose the light-weight Volumetric Transformer Block (VTB) by introducing identity V mapping, Multi-Scale Residual Path and Split-Head MLP. Then, we furthermore reduce the overall computational cost by applying cross-scale weight sharing on them. VTBs are specially designed, and they can share their weights even when having a different size input. Cross-scale weight sharing enhances VTBs cross-scale modeling ability while reducing the computational cost at the same time.

