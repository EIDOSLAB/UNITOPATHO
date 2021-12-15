# UNITOPATHO 
## A Labeled Histopathological Dataset for Colorectal Polyps Classification and Adenoma Dysplasia Grading

Carlo Alberto Barbano<sup>1</sup>, Daniele Perlo<sup>1</sup>, Enzo Tartaglione<sup>1</sup>, Attilio Fiandrotti<sup>1</sup>, Luca Bertero<sup>2</sup>, Paola Cassoni<sup>2</sup>, Marco Grangetto<sup>1</sup> 
| [[pdf](https://ieeexplore.ieee.org/document/9506198)]


1<sub>University of Turin, Computer Science dept.</sub><br>
2<sub>University of Turin, Medical Sciences dept.</sub>
<br/>

![UniToPatho](assets/unitopatho.png)

*UniToPatho* is an annotated dataset of **9536** hematoxylin and eosin stained patches extracted from 292 whole-slide images, meant for training deep neural networks for colorectal polyps classification and adenomas grading. The slides are acquired through a Hamamatsu Nanozoomer S210 scanner at 20× magnification (0.4415 μm/px). Each slide belongs to a different patient and is annotated by expert pathologists, according to six classes as follows:


- **NORM** - Normal tissue;
- **HP** - Hyperplastic Polyp;
- **TA.HG** - Tubular Adenoma, High-Grade dysplasia;
- **TA.LG** - Tubular Adenoma, Low-Grade dysplasia;
- **TVA.HG** - Tubulo-Villous Adenoma, High-Grade dysplasia;
- **TVA.LG** - Tubulo-Villous Adenoma, Low-Grade dysplasia.


## Downloading the dataset

You can download UniToPatho from [IEEE-DataPort](https://ieee-dataport.org/open-access/unitopatho)

## Dataloader and example usage

We provide a [PyTorch compatible dataset class](/unitopatho.py) and [ECVL compatible dataloader](/unitopatho_ecvl.py).
For example usage see [Example.ipynb](/Example.ipynb)

## Citation

If you use this dataset, please make sure to cite the [related work](https://arxiv.org/abs/2101.09991):

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unitopatho-a-labeled-histopathological/colorectal-polyps-characterization-on)](https://paperswithcode.com/sota/colorectal-polyps-characterization-on?p=unitopatho-a-labeled-histopathological)

```
@INPROCEEDINGS{9506198,
  author={Barbano, Carlo Alberto and Perlo, Daniele and Tartaglione, Enzo and Fiandrotti, Attilio and Bertero, Luca and Cassoni, Paola and Grangetto, Marco},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={Unitopatho, A Labeled Histopathological Dataset for Colorectal Polyps Classification and Adenoma Dysplasia Grading}, 
  year={2021},
  volume={},
  number={},
  pages={76-80},
  doi={10.1109/ICIP42928.2021.9506198}
}
```
