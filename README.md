<div align="center">

# Cross-modal learning for SAR target recognition using optical vision foundation models

### Code for EO-to-SAR prototype alignment

[![arXiv](https://img.shields.io/badge/arXiv-2609.07753-b31b1b.svg)](https://arxiv.org/abs/2609.07753)



**[Lucas Hirsch](https://luhirsch.github.io/) · [James R. Hopgood](https://www.research.ed.ac.uk/en/persons/james-hopgood/) · Javid Khan · [Yoann Altmann](https://researchportal.hw.ac.uk/en/persons/yoann-altmann/) · [Mike E. Davies](https://eng.ed.ac.uk/about/people/professor-michael-e-davies)**


📝 [Paper](https://arxiv.org/abs/2609.07753) · Accepted for presentation at SPIE Sensors + Imaging 2026

</div>



> **Under construction:** this repository is being prepared for release.


This repository contains the code accompanying the conference paper
*Cross-modal learning for SAR target recognition using optical vision foundation models*.


We investigate whether electro-optical (EO) vision foundation models can
provide supervision for synthetic aperture radar (SAR) target
recognition. A frozen DINOv3 EO encoder is used to construct optical class
prototypes without requiring paired EO and SAR images. A SAR
encoder is then trained to classify SAR images while aligning its embeddings
with the corresponding EO class prototypes. At inference time, the model
operates using SAR imagery alone.
