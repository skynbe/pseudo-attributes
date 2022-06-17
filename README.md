
# Unsupervised Learning of Debiased Representations with Pseudo-Attributes
<b> Pytorch implementation for BPA (CVPR 2022) </b>

[Seonguk Seo](https://seoseong.uk/), [Joon-Young Lee](https://joonyoung-cv.github.io/), [Bohyung Han](https://cv.snu.ac.kr/~bhhan/)

Seoul National University, Adobe Research
### [[Paper](https://arxiv.org/abs/2108.02943)]

> <p align="center">  <figcaption align="center"><b></b></figcaption>
> Dataset bias is a critical challenge in machine learning since it often leads to a negative impact on a model due to the unintended decision rules captured by spurious correlations. Although existing works often handle this issue based on human supervision, the availability of the proper annotations is impractical and even unrealistic. To better tackle the limitation, we propose a simple but effective unsupervised debiasing technique. Specifically, we first identify pseudo-attributes based on the results from clustering performed in the feature embedding space even without an explicit bias attribute supervision. Then, we employ a novel cluster-wise reweighting scheme to learn debiased representation; the proposed method prevents minority groups from being discounted for minimizing the overall loss, which is desirable for worst-case generalization. The extensive experiments demonstrate the outstanding performance of our approach on multiple standard benchmarks, even achieving the competitive accuracy to the supervised counterpart.


---

## Installation
```
git clone https://github.com/skynbe/pseudo-attributes.git
cd pseudo-attributes
pip install -r requirements.txt
```
Download CelebA dataset at $ROOT_PATH/data/celebA.

  
### Quick Start 

Train baseline model:
```
python main.py --arch ResNet18 --trainer classify --desc base --dataset celebA --test_epoch 1 --lr 1e-4 --target_attr Blond_Hair --bias_attrs Male --no_save
```

Train BPA model:
```
python main.py --arch ResNet18 --trainer bpa --desc bpa_k8 --dataset celebA --test_epoch 1 --lr 1e-4 --target_attr Blond_Hair --bias_attrs Male --k 8 --ks 8 --no_save --use_base {$BASE_PATH}
```

                
## Citation

If you find our work useful in your research, please cite:

```
@inproceedings{seo2022unsupervised,
  title={Unsupervised Learning of Debiased Representations with Pseudo-Attributes},
  author={Seo, Seonguk and Lee, Joon-Young and Han, Bohyung},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16742--16751},
  year={2022}
}
```
