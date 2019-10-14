# SRNTT_Pytorch
Reimplementation of paper "[Image Super-Resolution by Neural Texture Transfer](http://web.eecs.utk.edu/~zzhang61/project_page/SRNTT/cvpr2019_final.pdf)" in CVPR 2019 by pytorch.

Raw code in tensorflow can be found [here](https://github.com/ZZUTK/SRNTT).

---

*offline_patchMatch_textureSwap.py has been uploaded. ~~However it consumes huge space on disk when generating Ref-DIV2K dataset (100MB per picture). I'll try to find out if there're some bugs in this script.~~*

Because there are so many channels at each scale, the size of npz file is actually this large. Under this condition, I can't store the whole dataset in my hard disk (~800GB left). So next I'll try to use part of DIV2K (with reference pic) or CUFED.

---

**This repo has not been finished yet.**
