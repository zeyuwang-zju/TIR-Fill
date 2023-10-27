# TIR-Fill - THERMAL INFRARED IMAGE INPAINTING VIA EDGE-AWARE GUIDANCE
Source Code - ''THERMAL INFRARED IMAGE INPAINTING VIA EDGE-AWARE GUIDANCE'' - ICASSP 2023

![image](https://github.com/zeyuwang-zju/TIR-Fill/assets/112078495/994f1711-a2da-4040-a2e9-a7ca396aa60f)

## ABSTRACT

Image inpainting has achieved fundamental advances with deep learning. However, almost all existing inpainting methods aim to process natural images, while few target Thermal Infrared (TIR) images, which have widespread applications. When applied to TIR images, conventional inpainting methods usually generate distorted or blurry content. In this paper, we propose a novel task—Thermal Infrared Image Inpainting, which aims to reconstruct missing regions of TIR images. Crucially, we propose a novel deep-learning-based model TIR-Fill. We adopt the edge generator to complete the canny edges of broken TIR images. The completed edges are projected to the normalization weights and biases to enhance edge awareness of the model. In addition, a refinement network based on gated convolution is employed to improve TIR image consistency. The experiments demonstrate that our method outperforms state-of-the-art image inpainting approaches on FLIR thermal dataset.  



## USAGE

To train our model, firstly prepare the FLIR dataset [1] and the irregular masks with arbitrary mask ratios provided by Liu et al. [2].

Then, run the training scripts in three steps:

```python
# step 1
bash scripts/FLIR/s1_extract_edge.sh
# step 2
bash scripts/FLIR/s2_train_edge.sh
# step 3
bash scripts/FLIR/s3_train_gen.sh   
```



[1] Marcelo Bertalmio, Guillermo Sapiro, Vincent Caselles, and Coloma Ballester, “Image inpainting,” in Proceedings of the 27th annual conference on Computer graphics and interactive techniques, 2000, pp. 417–424.  

[2] Guilin Liu, Fitsum A Reda, Kevin J Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro, “Image inpainting for irregular holes using partial convolutions,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 85–100 .



## EXPERIMENTS

![image](https://github.com/zeyuwang-zju/TIR-Fill/assets/112078495/6dae5e5e-2dca-404a-a623-e9321566ee4f)

![1698389716734](https://github.com/zeyuwang-zju/TIR-Fill/assets/112078495/422c5bbd-06a5-418b-b6af-2dd520bcba57)



## CITATION

If you are interested this repo for your research, welcome to cite our paper:

```
@inproceedings{wang2023thermal,
  title={Thermal Infrared Image Inpainting Via Edge-Aware Guidance},
  author={Wang, Zeyu and Shen, Haibin and Men, Changyou and Sun, Quan and Huang, Kejie},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
