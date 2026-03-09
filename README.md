# [In Progress] Unified Framework for SiT / REPA / DDT / DUPA

A unified research codebase for diffusion transformers, representation alignment, and condition alignment.

This repository integrates **SiT**, **REPA**, **DDT**, and **DUPA training** into a single experimental framework, with a focus on:

- fair comparison across architectures and alignment strategies,
- flexible training and inference pipelines,
- customizable representation-alignment objectives,
- gradient interaction analysis during and after training,
- reproducible DUPA-style experiments. (Note: DUPA isn't open-source until now!)


### Flexible REPA target definition
You can customize **what to align** and **where to align**:

- which **SiT / DiT / DDT block** to hook,
- which **DINO block** to use as supervision,

### Multiple alignment losses
The framework is designed to support multiple alignment objectives, including:

- **Cosine alignment loss**
- **Dispersive loss**
- **Attention-based alignment loss**
- **Self-similarity loss**
- **InfoNCE-L2 loss**
- **InfoNCE-cosine loss**

You can swap or combine them to study how different alignment signals affect optimization, stability, and generation quality.

### Gradient interaction analysis

The framework supports:
- logging gradient statistics **during training**,
- measuring **gradient cosine similarity or angle** between losses,
- computing block-wise and whole-model gradient norms,
- running **post-training checkpoint analysis** to inspect how gradient relations evolve across training.

### Visualize by PCA and t-SNE

The framework supports:
- Use **PCA** to analyze different blocks of SiT,
- Use **t-SNE** to analyze different blocks of SiT.
  
### Fairer comparison between DDT and REPA variants
Some implementation details differ across released codebases and may lead to unfair conclusions if copied directly.

This framework fixes and standardizes parts of the setup so that comparisons across:
- **DDT**,
- **REPA-enhanced DiT / SiT**,
- **DUPA-like variants**

are more controlled and fair. (Note: All settings are the same as REPA.)


## Usages
### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py fit -c configs/XXX.yaml
```

### Predicting/Sampling
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py predict -c configs/XXXyaml --ckpt_path=XXXX
```
**Saving Logic**： 
- eval_max_num_instances: 50 does not literally mean saving only 50 samples. In practice, the number of saved samples is rounded up to a larger value. For example, setting it to 50 will actually save 1000 samples.

- During sampling, the samples are generated on 8 GPUs with eval_batch_size = 32, so each round produces 32 × 8 = 256 samples in total. Therefore, the saving process is performed over multiple rounds until the target number of samples is reached, and the last round may contain fewer samples.

### Evaluation

To compare different generative models, we use FID, sFID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples, which we store in `.npz` (numpy) files.

#### Download batches

We provide pre-computed sample batches for the reference datasets, our diffusion models, and several baselines we compare against. These are all stored in `.npz` format.

Reference dataset batches contain pre-computed statistics over the whole dataset, as well as 10,000 images for computing Precision and Recall. All other batches contain 50,000 images which can be used to compute statistics and Precision/Recall.

Here are links to download all of the sample and reference batches:

 * LSUN
   * LSUN bedroom: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz)
     * [ADM (dropout)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/admnet_dropout_lsun_bedroom.npz)
     * [DDPM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/ddpm_lsun_bedroom.npz)
     * [IDDPM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/iddpm_lsun_bedroom.npz)
     * [StyleGAN](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/stylegan_lsun_bedroom.npz)
   * LSUN cat: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/cat/VIRTUAL_lsun_cat256.npz)
     * [ADM (dropout)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/cat/admnet_dropout_lsun_cat.npz)
     * [StyleGAN2](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/cat/stylegan2_lsun_cat.npz)
   * LSUN horse: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/horse/VIRTUAL_lsun_horse256.npz)
     * [ADM (dropout)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/horse/admnet_dropout_lsun_horse.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/horse/admnet_lsun_horse.npz)

 * ImageNet
   * ImageNet 64x64: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/admnet_imagenet64.npz)
     * [IDDPM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/iddpm_imagenet64.npz)
     * [BigGAN](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/biggan_deep_imagenet64.npz)
   * ImageNet 128x128: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/VIRTUAL_imagenet128_labeled.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/admnet_imagenet128.npz)
     * [ADM-G](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/admnet_guided_imagenet128.npz)
     * [ADM-G, 25 steps](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/admnet_guided_25step_imagenet128.npz)
     * [BigGAN-deep (trunc=1.0)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/biggan_deep_trunc1_imagenet128.npz)
   * ImageNet 256x256: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_imagenet256.npz)
     * [ADM-G](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_guided_imagenet256.npz)
     * [ADM-G, 25 step](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_guided_25step_imagenet256.npz)
     * [ADM-G + ADM-U](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_guided_upsampled_imagenet256.npz)
     * [ADM-U](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/admnet_upsampled_imagenet256.npz)
     * [BigGAN-deep (trunc=1.0)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/biggan_deep_trunc1_imagenet256.npz)
   * ImageNet 512x512: [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz)
     * [ADM](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_imagenet512.npz)
     * [ADM-G](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_guided_imagenet512.npz)
     * [ADM-G, 25 step](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_guided_25step_imagenet512.npz)
     * [ADM-G + ADM-U](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_guided_upsampled_imagenet512.npz)
     * [ADM-U](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/admnet_upsampled_imagenet512.npz)
     * [BigGAN-deep (trunc=1.0)](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/biggan_deep_trunc1_imagenet512.npz)

#### Run evaluations

First, generate or download a batch of samples and download the corresponding reference batch for the given dataset. For this example, we'll use ImageNet 256x256, so the refernce batch is `VIRTUAL_imagenet256_labeled.npz` and we can use the sample batch `admnet_guided_upsampled_imagenet256.npz`.

Next, run the `evaluator.py` script. The requirements of this script can be found in [requirements.txt](requirements.txt). Pass two arguments to the script: the reference batch and the sample batch. The script will download the InceptionV3 model used for evaluations into the current working directory (if it is not already present). This file is roughly 100MB.

The output of the script will look something like this, where the first `...` is a bunch of verbose TensorFlow logging:

```
$ python evaluator.py VIRTUAL_imagenet256_labeled.npz admnet_guided_upsampled_imagenet256.npz
...
computing reference batch activations...
computing/reading reference batch statistics...
computing sample batch activations...
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: 215.8370361328125
FID: 3.9425574129223264
sFID: 6.140433703346162
Precision: 0.8265
Recall: 0.5309
```


### Classifier-Free Guidance (CFG) Modification Guide
In this framework, the guidance scale is controlled in the config file:
- guidance.w = 1.0: without CFG
- guidance.w > 1.0: with CFG

When enabling CFG, the training pipeline also needs to be adjusted accordingly.

#### Without CFG
In src.diffusion.base.training:
```python
def _impl_trainstep(self, net, ema_net, raw_images, x, y, step):
    raise NotImplementedError

def __call__(self, net, ema_net, raw_images, x, condition, uncondition, step):
    return self._impl_trainstep(net, ema_net, raw_images, x, condition, step)
```
In this case, the training step directly uses the conditional input.

#### With CFG
When CFG is enabled, conditional and unconditional inputs should be jointly prepared before the actual training step:
```python
def _impl_trainstep(self, net, ema_net, raw_images, x, y, step):
    raise NotImplementedError

def __call__(self, net, ema_net, raw_images, x, condition, uncondition, step):
    raw_images, x, condition = self.preproprocess(raw_images, x, condition, uncondition)
    return self._impl_trainstep(net, ema_net, raw_images, x, condition, step)
```
This step prepares the inputs required for classifier-free guidance by incorporating both conditional and unconditional branches before calling _impl_trainstep.

### [1] Choose different yaml could help you adjust the training target
```yaml
  denoiser:
    class_path: src.models.denoiser.decoupled_sit.DDT
```
- **DUPA**: decoupled_sit.DUPA
- **DDT**: decouple_improved_dit.DDT
- **REPA**: repa_sit.REPA
- **SiT**: repa_sit.SiT

### [2] Choose different loss
In src/diffusion/stateful_flow_matching/training_repa.py

- **Cosine alignment loss** : cos_loss 
- **Dispersive loss** : disp_loss
- **Attention-based alignment loss** : In progress, release recently
- **Self-similarity loss** : In progress, release recently
- **InfoNCE-L2 loss** : info_nce_l2_loss
- **InfoNCE-cosine loss** : info_nce_cosine_loss

### [3] Gradient angle Logging
You can set any gradident angle(such as angle between cosine loss and flow matching loss). And considering encoder-decoder architecture in DDT, you need to adjust encoder and decoder anchor, which represents angle is computed on the anchor block. Note: encoder anchor blovk num must be less than alignment block number, decoder is also the same.

**For training angle**, gradient will be automatically logged via src/callbacks/save_grad_stats.py. However, sometimes this logging will be inaccurate because of small local batch. More accurately, **for after-training angle**, you can use test_cg to compute the gradient angle.
```bash
bash test.sh
```
Note that you should execute this command multiple times for each experiment's result ckpt folder (which may contain multiple folders, possibly 4 or 8), because the file paths and --gradient parameters passed in each time will result in different results.

e.g., for alignment 0 1 2 3 (order doesn't matter), locate the corresponding ckpt folder, usually something like exp_repa_fixt_xl-009. It will automatically find all ckpt files within it. Additionally, modify the --gradient parameter in line 40 below to 0 1 2 3 (1234 setting), 0 1 (alignment 0 1), 4 5 6 7 (alignment 4567), etc., to obtain different results.

**Tips**: For alignments of 4567 and 7654, both use the --gradient parameter as 4 5 6 7; the order doesn't matter.

### [4] Visualize

```bash
python main_pca.py --ckpt=/data/lzw_25/result_ckpt/SiT_XL_2/XXX.ckpt --save_path=/data/lzw_25/DDT_revise/pca_vis/XXX.png
```

Please revise ckpt path and save_path according to your needs.

### [5] Fair Comparison

The original DDT configuration uses different settings, such as mixed precision and numerical precision. We revised these settings to enable a fairer comparison.

### We plan to integrate more components, such as CKNNA, to further enrich the functionality of this repository.

## Reference
```bibtex
@article{DBLP:journals/corr/abs-2504-05741,
  author       = {Shuai Wang and
                  Zhi Tian and
                  Weilin Huang and
                  Limin Wang},
  title        = {{DDT:} Decoupled Diffusion Transformer},
  journal      = {CoRR},
  volume       = {abs/2504.05741},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2504.05741},
  doi          = {10.48550/ARXIV.2504.05741},
  eprinttype    = {arXiv},
  eprint       = {2504.05741},
  timestamp    = {Mon, 19 May 2025 13:58:56 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2504-05741.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
```bibtex
@article{DBLP:journals/corr/abs-2410-06940,
  author       = {Sihyun Yu and
                  Sangkyung Kwak and
                  Huiwon Jang and
                  Jongheon Jeong and
                  Jonathan Huang and
                  Jinwoo Shin and
                  Saining Xie},
  title        = {Representation Alignment for Generation: Training Diffusion Transformers
                  Is Easier Than You Think},
  journal      = {CoRR},
  volume       = {abs/2410.06940},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2410.06940},
  doi          = {10.48550/ARXIV.2410.06940},
  eprinttype    = {arXiv},
  eprint       = {2410.06940},
  timestamp    = {Tue, 19 Nov 2024 08:58:20 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2410-06940.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
```bibtex
@article{DBLP:journals/corr/abs-2401-08740,
  author       = {Nanye Ma and
                  Mark Goldstein and
                  Michael S. Albergo and
                  Nicholas M. Boffi and
                  Eric Vanden{-}Eijnden and
                  Saining Xie},
  title        = {SiT: Exploring Flow and Diffusion-based Generative Models with Scalable
                  Interpolant Transformers},
  journal      = {CoRR},
  volume       = {abs/2401.08740},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2401.08740},
  doi          = {10.48550/ARXIV.2401.08740},
  eprinttype    = {arXiv},
  eprint       = {2401.08740},
  timestamp    = {Thu, 01 Feb 2024 15:35:36 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2401-08740.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgement
The code is mainly built upon [DDT](https://github.com/MCG-NJU/DDT), we also borrow ideas from the [REPA](https://github.com/sihyun-yu/REPA), [DUPA](https://openreview.net/forum?id=ALpn1nQj5R) and [SiT](https://github.com/willisma/SiT).
