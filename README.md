# [ICCV 2025] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors

[**Project Page**](https://xiaokunsun.github.io/StrandHead.github.io) | [**Arxiv**](https://arxiv.org/abs/2412.11586) | [**Gallery**](https://drive.google.com/drive/folders/1Ve2vVVilzI-2TYNB9wQrLgG53L2PjFBM?usp=sharing)

Official repo of "StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors"

[Xiaokun Sun](https://xiaokunsun.github.io), [Zeyu Cai](https://github.com/zcai0612), [Ying Tai](https://tyshiwo.github.io/index.html), [Jian Yang](https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ), [Zhenyu Zhang](https://jessezhang92.github.io)

<p align="center"> All Code will be released soon... 🚀🚀🚀 </p>

Abstract: While haircut indicates distinct personality, existing avatar generation methods fail to model practical hair due to the data limitation or entangled representation. We propose StrandHead, a novel text-driven method capable of generating 3D hair strands and disentangled head avatars with strand-level attributes. Instead of using large-scale hair-text paired data for supervision, we demonstrate that realistic hair strands can be generated from prompts by distilling 2D generative models pre-trained on human mesh data. To this end, we propose a meshing approach guided by strand geometry to guarantee the gradient flow from the distillation objective to the neural strand representation. The optimization is then regularized by statistically significant haircut features, leading to stable updating of strands against unreasonable drifting. These employed 2D/3D human-centric priors contribute to text-aligned and realistic 3D strand generation. Extensive experiments show that StrandHead achieves the state-of-the-art performance on text to strand generation and disentangled 3D head avatar modeling. The generated 3D hair can be applied on avatars for strand-level editing, as well as implemented in the graphics engine for physical simulation or other applications.

<p align="center">
    <img src="assets/teaser.png">
</p>

## BibTeX

```bibtex
@inproceedings{StrandHead,
  title={StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors},
  author={Sun, Xiaokun and Cai, Zeyu and Tai, Ying and Yang, Jian and Zhang, Zhenyu},
  booktitle=ICCV,
  year={2025}
}
```
