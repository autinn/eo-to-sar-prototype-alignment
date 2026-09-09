<div align="center">
# Cross-modal learning for SAR target recognition using optical vision foundation models
## Code for  EO-to-SAR Prototype Alignment

<!-- [![Score](https://img.shields.io/badge/Place-3rd-brown)]() -->
[![Paper](https://img.shields.io/badge/Paper-Link-blue?style=plastic)](https://arxiv.org/abs/2609.07753)

🎓 **Authors**: [Lucas Hirsch](https://luhirsch.github.io/), [Mike Davies](https://eng.ed.ac.uk/about/people/professor-michael-e-davies)

🏫 **Affiliation**: Institute for Imaging, Data and Communications (IDCOM), School of Engineering, University of Edinburgh, UK

📝 **Paper:** Pre-print available on  [arXiv](https://arxiv.org/abs/2609.07753). The paper has been accepted for publication at SPIE Sensors + Imaging 2026.
</div>


> **Under construction:** this repository is being prepared for release.


This repository contains the code for the SPIE conference paper *Cross-Modal
Learning for SAR Target Recognition Using Optical Vision Foundation
Models*. 

The project studies how electro-optical (EO) vision foundation models
can provide class level prototype references for Synthetic Aperture Radar (SAR)
target recognition. A frozen EO DINOv3 encoder is used to construct optical
class prototypes, while a SAR encoder (fine tuned with LoRA) is trained to classify SAR
images and align its embeddings to the corresponding EO prototypes. At inference
time, the model uses only SAR imagery.
