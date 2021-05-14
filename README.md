# Robust unlearnable examples

The tremendous amount of accessible data on the internet face the risk of being unauthorized used for training machine learning models. To address such a privacy concern, \citet{huang2021unlearnable} propose to generate {\it error-minimizing noise}, which can make training examples unlearnable from machine learning models. However, it is found that such unlearnability can be easily broken via adversarial training, which makes the previous privacy-preserving effort in vain. In this paper, we show that training examples can also be made unlearnable against adversarial training. {\color{red}We first analyze the generation of the error-minimizing noise and find that the effect of error-minimizing noise is to modify the distribution of the training examples such that the modified distribution is significantly different from the original distribution but can be easily fit by machine learning models.} Based on this insight, we then propose to generate {\it robust error-minimizing noise}, which aims to make the training examples to be easily fit by adversarial training. Specifically, we propose a novel min-min-max training framework to generate such robust error-minimizing noise. We also adopt the technique for generating robust adversarial examples into our framework to further enhance the robustness of our robust error-minimizing noise against data augmentation. Experiments on CIFAR-10 and CIFAR-100 datasets show that our robust error-minimizing noise is effective, and is transferable across different model architectures. The source code package has been submitted and will be released publicly.

## Code structures

```
|---- ./
    |---- attacks/
        |---- __init__.py
        |---- pgd.py
        |---- minimax_pgd_defender.py
        |---- robust_workers.py
    |---- models/
        |---- __init__.py
        |---- resnet.py
        |---- vgg.py
    |---- utils/
        |---- __init__.py
        |---- argument.py
        |---- data.py
        |---- generic.py
    |---- perturbation.py
    |---- robust_minimax_pgd_perturbation.py
    |---- robust_noise_generation_v22.py
    |---- eval.py
    |---- train.py
```

## Tasks

- Architectures: VGG-16, ResNet-18, DenseNet-121, Wide ResNet-34-10.
- Datasets: CIFAR-10, CIFAR-100, ~~Tiny-ImageNet, ImageNet~~.
- Adv training methods: PGD [2], ~~CW [3], L2L [4]~~.

| Tasks                                                        | Dates |
| ------------------------------------------------------------ | ----- |
| Reproduce the original unlearnable examples experiments.     |       |
| Implement and evaluate the adversarial training-based unlearnability breaker. |       |
| Implement and evaluate the min-max-min framework **with PGD.** |       |
| Add CW (easy task) and L2L (hard task).                      |       |
| TBD                                                          |       |

## References

[1] Huang, H., Ma, X., Erfani, S. M., Bailey, J., & Wang, Y. (2021). Unlearnable Examples: Making Personal Data Unexploitable. *arXiv preprint arXiv:2101.04898*.

[2] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. *arXiv preprint arXiv:1706.06083*.

[3] Carlini, N., & Wagner, D. (2017, May). Towards evaluating the robustness of neural networks. In *2017 ieee symposium on security and privacy (sp)* (pp. 39-57). IEEE.

[4] Jiang, H., Chen, Z., Shi, Y., Dai, B., & Zhao, T. (2018). Learning to defense by learning to attack. *arXiv preprint arXiv:1811.01213*.