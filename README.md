# Unsupervised Domain Adaptation for Medical Image Segmentation on AMOS-22

*Senior Design Project, Department of Electrical and Computer Engineering, North South University.*

> Markdown conversion of `report_main.tex`. Citation keys are kept in pandoc syntax (`[@key]`) so they still resolve against `references.bib`. The two methodology diagrams are drawn in TikZ and exist only in the LaTeX source; their captions and numbering are preserved here.

---

![logo](figures/north-south-university-logo-png_seeklogo-351576.png)

**Department of Electrical and Computer Engineering**

**North South University**

**Senior Design Project**

**Unsupervised Domain Adaptation for
Medical Image Segmentation on
AMOS-22**

by

**MD. Shaiful Alam 2132537642**

**Nahid Sarwar Ratul 2212790042**

**MD. Jarif Mehtab Hossain 2111216642**

**Abu Ahmad 2121725042**

**Faculty Advisor:**

**Dr. Mohammad Abdul Qayum**

Assistant Professor

ECE Department

**Summer, 2026**

# APPROVAL

MD. Shaiful Alam (2132537642), Nahid Sarwar Ratul (2212790042), MD. Jarif Mehtab Hossain (2111216642) and Abu Ahmad (2121725042) from the Electrical and Computer Engineering Department of North South University have worked on the Senior Design Project titled “Unsupervised Domain Adaptation for Medical Image Segmentation on AMOS-22” under the supervision of Dr. Mohammad Abdul Qayum in partial fulfillment of the requirement for the degree of Bachelor of Science in Engineering and it has been accepted as satisfactory.

**Supervisor’s Signature**
…………………………
**Dr. Mohammad Abdul Qayum**
**Assistant Professor**
Department of Electrical and Computer Engineering
North South University, Dhaka, Bangladesh.

**Chairman’s Signature**
…………………………
**Dr. Mohammad Abdul Matin**
**Professor & Chair**
Department of Electrical and Computer Engineering
North South University, Dhaka, Bangladesh.

# DECLARATION

This is to declare that this project entitled, **“Unsupervised Domain Adaptation for Medical Image Segmentation on AMOS-22”** is our original work. No part of this work has been submitted elsewhere partially or fully for the award of any other degree or diploma. All project-related information will remain confidential and shall not be disclosed without the formal consent of the project supervisor. Relevant previous works presented in this report have been properly acknowledged and cited. The plagiarism policy, as stated by the supervisor, has been maintained.

**1. MD. Shaiful Alam**

------------------------------------------------------------------------


**2. Nahid Sarwar Ratul**

------------------------------------------------------------------------


**3. MD. Jarif Mehtab Hossain**

------------------------------------------------------------------------


**4. Abu Ahmad**

------------------------------------------------------------------------


Department of Electrical and Computer Engineering

North South University

**Date:** September 07, 2026

# ACKNOWLEDGEMENTS

The authors would like to express their heartfelt gratitude towards their project and research supervisor, Dr. Mohammad Abdul Qayum, Department of Electrical and Computer Engineering, North South University, Bangladesh, for his invaluable support, precise guidance and advice pertaining to the experiments, research and theoretical studies carried out during the course of the current project and also in the preparation of the current report.

Furthermore, the authors would like to thank the Department of Electrical and Computer Engineering, North South University, Bangladesh for facilitating the research. The authors would also like to thank their friends and peers for their assistance throughout this project.
MD, Shaiful Alam, Nahid Sarwar Ratul, MD. Jarif Mehtab Hossain, Abu Ahmad

North South University

**Date:** September 07, 2026

# ABSTRACT

**Unsupervised Domain Adaptation for Medical Image Segmentation on AMOS-22**

Deep learning segments abdominal organs accurately, but a model trained on one imaging modality usually fails on another. This project studies that failure, known as domain shift, between Computed Tomography (CT) and Magnetic Resonance Imaging (MRI) using the AMOS-22 abdominal multi-organ dataset. A three-dimensional convolutional segmentation network was trained only on CT scans with standard Hounsfield unit windowing and then evaluated on MRI scans to measure the gap between the two modalities. To reduce that gap without retraining any network parameter and without any target-domain label, a two-step unsupervised domain adaptation strategy was applied at inference time. First, dynamic $0.5$–$99.5$ percentile intensity scaling mapped non-standardised MRI signal intensities into a $[0,1]$ range. Second, Adaptive Batch Normalization froze all learnable weights and recalibrated the running mean and variance statistics of the normalization layers on unlabelled target MRI patches. In-domain CT performance served as an oracle upper bound for the same network. Naive cross-modality transfer performs poorly, reaching a mean Dice of only $0.495$ on MRI; percentile normalization alone raises this to $0.642$; and adding Adaptive Batch Normalization raises it further to $0.721$, against an oracle upper bound of approximately $0.80$. The adaptation therefore recovers much of the lost performance at inference time, on a single consumer graphics processing unit, while a considerable gap to the upper bound remains.

**Keywords:** Unsupervised Domain Adaptation, Cross-Modality Segmentation, Adaptive Batch Normalization, Percentile Intensity Normalization, Domain Shift, AMOS-22.

# Chapter 1: Introduction

## 1.1 Background and Motivation

Three-dimensional segmentation of organs from medical images is a core step in many clinical workflows, including radiotherapy planning, computer-aided diagnosis, surgical navigation, and quantitative disease monitoring. In radiotherapy, the precise delineation of organs-at-risk such as the kidneys, liver, and bladder determines how much radiation dose can be safely delivered to a tumour while sparing healthy tissue, so even small segmentation errors can change the quality of the treatment. In diagnostic radiology, organ volumes and shapes derived from segmentation support the detection and staging of disease, the measurement of tumour burden, and the longitudinal tracking of a patient’s response to therapy. Manual delineation of organs such as the liver, spleen, kidneys, and pancreas is slow, labour-intensive, and subject to considerable inter-observer and intra-observer variability; a single abdominal scan may contain hundreds of two-dimensional slices, and contouring every organ on every slice can take a trained radiologist a substantial part of an hour. Those costs, together with the growing volume of imaging data produced by modern hospitals, have driven extensive research into automated segmentation with deep learning [@litjens2017survey]. Convolutional neural networks, in particular encoder–decoder architectures such as the U-Net [@ronneberger2015unet] and its three-dimensional extension [@cicek20163dunet], are now the standard tools for the task. They learn the hierarchical features that distinguish one organ from another directly from labelled examples, and they approach expert-level accuracy on well-curated datasets.

Two imaging modalities dominate abdominal imaging: Computed Tomography (CT) and Magnetic Resonance Imaging (MRI). CT measures the attenuation of X-rays and produces images whose intensities are expressed in standardised Hounsfield units, which makes CT fast, inexpensive, and highly reproducible across scanners, at the cost of exposing the patient to ionising radiation. MRI instead exploits the magnetic properties of hydrogen nuclei and offers better soft-tissue contrast without any ionising radiation, but its intensity values are not standardised and depend heavily on the acquisition sequence, scanner, and vendor. Because the two images are formed in different ways, the statistical distribution of voxel intensities in a CT scan bears little resemblance to that of an MRI scan of the same anatomy: CT intensities are calibrated and cluster tightly around the Hounsfield value of air near $-1000$, whereas MRI intensities are arbitrary and spread broadly around zero. The practical consequence is that a segmentation model trained on one modality performs poorly, often catastrophically, when applied to the other, because the features it has learned to associate with each organ no longer match the intensities it encounters [@guan2021survey]. This phenomenon, known as *domain shift*, is one of the main obstacles preventing deep learning segmentation models from being deployed reliably across the varied scanners, protocols, and modalities found in real hospitals. A model that is accurate in the laboratory on the modality it was trained on may silently produce unusable segmentations once it encounters data from a different source, which is a serious concern for clinical safety.

Collecting and manually annotating a large labelled dataset for every modality and every scanner would remove the problem, but expert annotation is the single most costly resource in medical image analysis, so that route is rarely affordable. *Domain adaptation* techniques exist for this reason: they transfer the knowledge captured by a model trained on a labelled source modality to an unlabelled target modality, and so avoid a second round of expensive annotation. Where the adaptation works reliably, a hospital could train a model once on its plentiful CT archives and deploy it on MRI, or the reverse, without re-annotating a single new scan.

The release of the AMOS-22 benchmark [@ji2022amos], which provides 500 CT and 100 MRI abdominal scans with voxel-level annotations of 15 organs, and the recent M3DA benchmark for unsupervised domain adaptation in three-dimensional medical images [@shirokikh2025m3da], have made it possible to study cross-modality domain shift rigorously on large, public, and clinically diverse data. Earlier domain adaptation work was largely evaluated on small or private datasets, which made fair comparison difficult and left open how well any given method would generalise. These benchmarks close that gap and motivate the present project, which investigates the CT-to-MRI domain gap and evaluates a lightweight, resource-efficient adaptation strategy suitable for training on commodity hardware rather than a large computing cluster.

## 1.2 Purpose and Goal of the Project

The purpose of this project is to quantify and reduce the performance gap that arises when a three-dimensional abdominal organ segmentation model trained on CT is applied to MRI. The specific goals and contributions are:

- **Quantify the gap.** Measure, organ by organ, how far a CT-trained three-dimensional model drops when it is applied to MRI without any adaptation, which sets a naive lower bound under severe domain shift.

- **Adapt at test time.** Recover performance without retraining any network parameter and without any target-domain label, using a two-step strategy that combines dynamic percentile intensity normalization with Adaptive Batch Normalization [@li2018adabn].

- **Build efficiently.** Implement and run the complete three-dimensional pipeline end-to-end on a single consumer graphics processing unit, so that the study can be reproduced without access to a computing cluster.

- **Benchmark.** Compare the naive, adapted, and oracle settings on a held-out MRI set, following the experimental philosophy of the M3DA benchmark [@shirokikh2025m3da], and report the fraction of the domain gap actually recovered rather than a bare score.

## 1.3 Organization of the Report

The remainder of this report is organised as follows. Chapter 2 summarises the key concepts and the scope of the broader field a reader needs in order to follow the report. Chapter 3 reviews the relevant literature on medical image segmentation and domain adaptation, and identifies the limitations that motivate this work. Chapter 4 describes the methodology, including the system design, the software components, and the implementation of the pipeline. Chapter 5 presents the experiments, results, and a discussion of the findings. Chapter 6 examines the societal, health, and environmental impacts of the project. Chapter 7 outlines the project planning and budget. Chapter 8 discusses the complex engineering problems and activities addressed. Finally, Chapter 9 summarises the project, notes its limitations, and suggests directions for future improvement.

# Chapter 2: Background

## 2.1 Foundation

### 2.1.1 Volumetric Medical Image Segmentation

Semantic segmentation assigns a class label to every voxel in a scan. For a volume of size $H \times W \times D$, the model outputs a label map over $C+1$ classes, where the additional class is background. In this project $C = 15$, corresponding to the fifteen abdominal organs annotated in AMOS-22 [@ji2022amos], giving sixteen output channels in total.

The dominant architecture is the U-Net [@ronneberger2015unet]: an encoder that repeatedly convolves and downsamples to build an abstract, low-resolution representation, and a decoder that upsamples back to full resolution, with skip connections carrying fine spatial detail across so that boundaries survive the bottleneck. The three-dimensional variant [@cicek20163dunet] replaces every two-dimensional operation with its volumetric equivalent, which lets the network exploit through-plane context that a slice-wise model cannot see. The cost is memory: a full abdominal volume does not fit on a consumer graphics processing unit, so training operates on cropped patches of $96^3$ voxels, and full volumes are reconstructed at test time by sliding-window inference, tiling the volume with overlapping patches and aggregating their predictions.

Training minimises a combined Dice and cross-entropy objective [@milletari2016vnet]. The Dice term optimises region overlap directly and tolerates the severe foreground-to-background imbalance of abdominal scans, in which organs occupy a small fraction of the volume; the cross-entropy term supplies a well-behaved per-voxel gradient. The same Dice similarity coefficient, defined for a predicted region $P$ and a reference region $G$ as $$\mathrm{DSC}(P, G) \;=\; \frac{2\,|P \cap G|}{|P| + |G|},$$ also serves as the evaluation metric. It ranges from $0$ to $1$ and is the standard overlap measure in this field [@taha2015metrics].

### 2.1.2 Image Formation in CT and MRI

CT measures the attenuation of X-rays and reports it in Hounsfield units, a physically calibrated scale fixed by definition at $-1000$ for air and $0$ for water. Because the scale is absolute, a given tissue produces approximately the same value on any scanner, and a fixed intensity window ($[-175, 250]$ HU for abdominal soft tissue) is a meaningful and transferable preprocessing step.

MRI instead exploits the magnetic resonance of hydrogen nuclei. It offers markedly better soft-tissue contrast and involves no ionising radiation, but its intensities carry no absolute physical meaning: values depend on the pulse sequence, the field strength, the vendor, and even the receive coil, so the same organ imaged twice may occupy entirely different numeric ranges. This asymmetry is the root of the problem studied here. A fixed Hounsfield window applied to MRI is meaningless, and any method transferring a CT model to MRI must first establish a common intensity representation.

### 2.1.3 Domain Shift and Unsupervised Domain Adaptation

Let the source domain provide labelled pairs drawn from a distribution $P_s(x, y)$ and the target domain provide images drawn from $P_t(x)$. Domain shift is the condition $P_s(x) \neq P_t(x)$: the marginal distribution of image intensities differs, while the underlying anatomy and the labelling convention are shared. A model fitted to source features therefore encounters inputs it was never calibrated for, and accuracy degrades even though the task itself is unchanged [@guan2021survey].

Unsupervised domain adaptation (UDA) seeks good target performance using labelled source data together with *unlabelled* target images only. The principal families are:

- **Image-level translation**, which synthesises target-like images from source ones, typically with a CycleGAN [@zhu2017cyclegan], and trains a segmenter on the synthetic pairs.

- **Feature-level alignment**, which forces the encoder to produce domain-invariant features, usually through an adversarial discriminator [@ganin2015dann; @chen2020sifa].

- **Self-training and entropy minimisation**, which generate pseudo-labels on target data or directly sharpen the model’s target predictions [@vu2019advent].

- **Normalization and test-time adaptation**, which leave the learned weights untouched and correct only the intensity or activation statistics [@li2018adabn; @karani2021test; @wang2022sourcefree].

This project belongs to the last family, which is by far the cheapest: no second network, no adversarial optimisation, and no retraining, so it can be applied to a model that has already been deployed.

Interpreting any UDA result needs two reference quantities. The *baseline* is the source model applied to the target with no adaptation, and the *oracle* is the same architecture evaluated in the absence of domain shift. The interval between them is the domain gap, and the informative way to report an adaptation method is the fraction of that interval it recovers, rather than a raw score that conceals how large the gap was in the first place [@shirokikh2025m3da].

Adaptive Batch Normalization (AdaBN) [@li2018adabn] is the specific mechanism used in this project. A batch normalization layer [@ioffe2015batchnorm] maintains two kinds of state: learned affine parameters, and running estimates of the per-channel mean $\mu$ and variance $\sigma^2$ of its inputs. AdaBN freezes every learned parameter and re-estimates only the running statistics by passing unlabelled target images forward. No gradient is computed and nothing is learned, yet because those buffers summarise the activation distribution at each depth, replacing source estimates with target estimates re-centres and re-scales features throughout the network. This also explains an architectural constraint: a backbone built on instance normalization [@ulyanov2016instance] keeps no running statistics and would leave AdaBN nothing to recalibrate.

## 2.2 Context and Scope

Automated organ segmentation supports several parts of quantitative medical image analysis: radiotherapy planning, in which the delineation of organs-at-risk determines how much dose can be safely delivered; surgical navigation; and longitudinal disease monitoring. Manual contouring of a single abdominal scan can occupy a substantial part of a radiologist’s hour and varies between and within observers, so automation carries real clinical and economic value [@litjens2017survey].

The field has advanced quickly. Self-configuring pipelines such as nnU-Net [@isensee2021nnunet], transformer-based backbones such as UNETR [@hatamizadeh2022unetr] and Swin UNETR [@tang2022swinunetr], and promptable foundation models such as MedSAM [@ma2024medsam] now approach expert accuracy when training and test data are drawn from the same source. In-domain accuracy is largely solved; transfer is not. Hospitals operate a heterogeneous mix of scanners, vendors, protocols, and modalities, and a model validated in the laboratory can fail silently once deployed on data from a different source, which is a safety concern as much as a performance one.

Two recent benchmarks made this measurable. AMOS-22 [@ji2022amos] released 500 CT and 100 MRI abdominal scans from multiple centres, vendors, and imaging phases, each annotated for fifteen organs, although the released MRI subset omits bladder and prostate/uterus annotations and therefore supports evaluation over thirteen. The M3DA benchmark [@shirokikh2025m3da] then evaluated more than ten adaptation methods across eight realistic shifts and reported that even the best closes only approximately 62% of the domain gap on average, so the problem remains open and partial recovery is the realistic expectation.

The practical drivers are equally clear. Annotating a new modality is the single most expensive resource in this field. Privacy and governance rules frequently prevent a deployed model from being retrained against its original training archive [@wang2022sourcefree]. In settings with mixed, ageing scanner fleets and scarce annotation capacity, a method that adapts an existing model without labels or retraining is considerably more deployable than one requiring a fresh labelled dataset for every site.

This project is deliberately narrow in scope. It studies one shift, CT to MRI, on one dataset, AMOS-22, using a single three-dimensional U-Net trained on CT and adapted at inference by percentile intensity normalization followed by AdaBN. Performance is reported as Dice over the thirteen organs annotated in the MRI subset, measured against an explicit in-domain oracle, on a single consumer graphics processing unit.

What falls outside that scope is equally deliberate. No target label is used at any stage and no weight is updated after source training, which rules out supervised fine-tuning and gradient-based test-time adaptation. Adversarial image translation was attempted but not adopted: a two-dimensional CycleGAN [@zhu2017cyclegan] produced anatomically faithful translations, yet the downstream two-dimensional segmenter under-fit at the available data scale. Only the CT-to-MRI direction is studied, evaluation uses the Dice coefficient alone without boundary-based distances [@taha2015metrics], and the training schedule is shorter than those used in the original benchmarks, so absolute accuracy sits below the state of the art even where the relative trends are faithful.

# Chapter 3: Research Literature Review

## 3.1 Existing Research and Limitations

**Deep learning architectures for medical image segmentation.** The U-Net [@ronneberger2015unet] introduced the symmetric encoder–decoder design with skip connections that has become the foundation of medical image segmentation, and its three-dimensional variant [@cicek20163dunet] extended it to volumetric data. Milletari et al. [@milletari2016vnet] proposed the V-Net together with the Dice loss, which directly optimises overlap and handles class imbalance common in medical scans. Zhou et al. [@zhou2019unetpp] redesigned the skip connections in UNet++ to fuse multi-scale features, while the Attention U-Net [@oktay2018attention] added attention gates that suppress irrelevant background responses, which is beneficial for small abdominal organs. He et al. [@he2016resnet] introduced residual connections, and Ioffe and Szegedy [@ioffe2015batchnorm] proposed batch normalization; both are widely incorporated into segmentation backbones to stabilise and accelerate training of deep networks. Batch normalization matters here because the running mean and variance statistics it accumulates are the quantities that this project’s adaptation method recalibrates on the target domain. Myronenko [@myronenko2018segresnet] showed with SegResNet that an autoencoder regularisation branch improves volumetric segmentation under limited data. More recently, transformer-based models such as UNETR [@hatamizadeh2022unetr], Swin UNETR [@tang2022swinunetr], TransUNet [@chen2021transunet], CoTr [@xie2021cotr], and nnFormer [@zhou2023nnformer] have combined self-attention with convolutional decoders to model long-range context, and MedNeXt [@roy2023mednext] revisited pure convolutional scaling to match transformer accuracy. A unifying practical contribution is nnU-Net [@isensee2021nnunet], a self-configuring pipeline that automatically adapts pre-processing, architecture, and training to a given dataset and remains a strong baseline across many benchmarks; its use of percentile-based intensity clipping is the direct precedent for the normalization step adopted in this project. Foundation models have also reached medical imaging: the Segment Anything Model (SAM) [@kirillov2023sam] and its medical adaptation MedSAM [@ma2024medsam] offer promptable, general-purpose segmentation, while self-supervised pre-training such as Models Genesis [@zhou2021modelsgenesis] learns transferable 3D representations from unlabelled scans. The MONAI framework [@cardoso2022monai] provides open-source, reproducible implementations of many of these components and is the basis of the pipeline used in this project.

**Benchmarks and datasets for abdominal organ segmentation.** Progress has historically been limited by small, single-centre datasets. Early multi-organ efforts such as the Multi-Atlas Labeling Beyond the Cranial Vault (BTCV) challenge [@landman2015btcv] and the Medical Segmentation Decathlon [@antonelli2022msd] provided standard, if modest, CT benchmarks. The FLARE challenge [@ma2022flare] emphasised fast, low-memory abdominal CT organ segmentation, a goal aligned with the resource-efficient design of this project. For the cross-modality setting specifically, the CHAOS challenge [@kavur2021chaos] released paired CT and MR abdominal data, and the Multi-Modality Whole Heart Segmentation (MM-WHS) challenge [@zhuang2019mmwhs] became the classic CT-to-MR adaptation test-bed in cardiac imaging. The AMOS-22 benchmark [@ji2022amos] extended this line by releasing 500 CT and 100 MRI scans from multiple centres, vendors, phases, and diseases, each annotated for 15 organs, and by benchmarking several state-of-the-art models. AMOS-22 is now a standard test-bed for robust and cross-modality segmentation and is the dataset used in this project.

**Domain shift and domain adaptation.** Ganin and Lempitsky [@ganin2015dann] introduced domain-adversarial training to learn features that are invariant across domains, a principle later applied to structured segmentation outputs by Tsai et al. [@tsai2018adaptseg]. In medical imaging, Kamnitsas et al. [@kamnitsas2017unsupervised] demonstrated unsupervised adaptation for brain lesion segmentation with adversarial networks, building on their multi-scale 3D CNN [@kamnitsas2017deepmedic]. For the harder cross-modality setting, Dou et al. [@dou2018pnp] adapted convolutional networks between CT and MRI with an adversarial plug-and-play scheme, and Chen et al. [@chen2019synergistic; @chen2020sifa] proposed SIFA, which performs synergistic image and feature alignment bidirectionally. Image-translation approaches based on CycleGAN [@zhu2017cyclegan], CyCADA [@hoffman2018cycada], and SynSeg-Net [@huo2019synseg] synthesise target-like images to train a segmentation model without target labels, while ADVENT [@vu2019advent] instead minimises the entropy of target predictions to align domains. Ouyang et al. [@ouyang2019dsa] showed that data-efficient adaptation is possible by aligning latent distributions with a variational prior, and Fourier Domain Adaptation [@yang2020fda] aligns low-frequency amplitude spectra to bridge appearance gaps without any network for translation. A complementary line of work avoids target data altogether: domain generalization methods such as deep stacked transformation [@zhang2020generalizing], MixStyle [@zhou2021mixstyle], and causality-inspired augmentation [@ouyang2022gin] train only on the source domain but craft augmentations or feature perturbations that force modality-invariant features. This project belongs to the normalization-based family, which is attractive for its simplicity: instance normalization [@ulyanov2016instance] removes contrast-specific statistics, and adaptive batch normalization (AdaBN) [@li2018adabn] re-estimates the normalization statistics on the target domain while leaving every learned weight untouched. Test-time adaptation [@karani2021test], self-ensembling [@perone2019unsupervised], and source-free adaptation [@wang2022sourcefree] adapt models using only unlabelled target images (and, in the source-free case, without access to the source data), which is increasingly important under clinical privacy constraints. Comprehensive surveys of these families are given by Guan and Liu [@guan2021survey] and Litjens et al. [@litjens2017survey]. Most recently, the M3DA benchmark [@shirokikh2025m3da] evaluated more than ten adaptation methods across eight realistic shifts and reported that even the best method closes only about 62% of the domain gap on average, so the problem is far from solved.

**Optimisation, normalization, and evaluation.** The training in this project relies on the AdamW optimiser with decoupled weight decay [@loshchilov2019adamw], a combined Dice and cross-entropy loss [@milletari2016vnet], and batch normalization [@ioffe2015batchnorm] in the segmentation backbone, the last of these being a prerequisite for the AdaBN step [@li2018adabn] applied at inference. For evaluation, Taha and Hanbury [@taha2015metrics] provide the standard analysis of overlap-based and boundary-based metrics for three-dimensional medical segmentation; the Dice similarity coefficient adopted throughout this report is the overlap metric from that analysis, and the boundary-based distances they recommend are identified in Section 9.3 as a natural extension of the present evaluation.

**Limitations of existing work.** Four limitations recur in this literature. (i) Many cross-modality adaptation methods rely on adversarial image translation, which is difficult to train, computationally expensive, and often requires multiple networks, making it impractical on modest hardware. (ii) A large fraction of studies are evaluated on small or private datasets, which limits reproducibility and fair comparison, a gap that AMOS-22 [@ji2022amos] and M3DA [@shirokikh2025m3da] were created to close. (iii) Many methods require access to the source training data, or a full retraining cycle, at adaptation time, which is often impossible once a model has been deployed in a hospital and the training archive is no longer accessible [@wang2022sourcefree]. (iv) The comparatively simple normalization-based adaptations are frequently reported only as ablation rows inside larger systems, so how much of the gap they close on their own is rarely stated plainly. These observations motivate the present project, which implements a lightweight, non-adversarial, inference-time adaptation that requires neither target labels nor a second network nor any retraining, evaluates it on the public AMOS-22 dataset, and reports the recovery it achieves against an explicit oracle upper bound.

# Chapter 4: Methodology

## 4.1 System Design

The system follows a classical source-to-target domain adaptation design. CT is treated as the labelled *source* modality and MRI as the unlabelled *target* modality. Formally, the task is voxel-wise semantic segmentation: given a three-dimensional image volume $x \in \mathbb{R}^{H \times W \times D}$, the model must predict a label map $y \in \{0, 1, \dots, C\}^{H \times W \times D}$, where $C = 15$ is the number of foreground organs and the value $0$ denotes background, giving 16 output classes in total. The source domain provides a set of labelled pairs $\{(x_i^s, y_i^s)\}$ drawn from the CT distribution $P_s(x, y)$, whereas the target domain provides images $\{x_j^t\}$ drawn from a different MRI distribution $P_t(x)$. Domain shift means that $P_s(x) \neq P_t(x)$: the marginal distributions of image intensities differ between the two modalities, even though the underlying anatomy and its labelling convention are shared. The goal of domain adaptation is to obtain a segmentation function that performs well on the target distribution despite having been trained only on source data.

Adaptation here happens entirely *at inference time*. No target label is ever used, no network weight is ever updated after source training, and no second network is trained. This makes the method applicable to an already-deployed model and keeps the computational cost negligible relative to adversarial alternatives.

A four-stage experimental pipeline was established, illustrated in Figure 1:

1.  **Source.** A baseline three-dimensional U-Net is trained exclusively on CT volumes using standard Hounsfield unit windowing.

2.  **Naive.** The CT-trained network is applied directly to unadapted target MRI scans. This establishes the naive lower-bound performance under severe domain shift, and represents what a practitioner would obtain by pointing an existing CT model at MRI data.

3.  **Adapt.** A two-step unsupervised adaptation is applied during inference. Dynamic $0.5$–$99.5$ percentile intensity scaling first aligns the non-standardised MRI signal intensities to a $[0,1]$ range. Adaptive Batch Normalization (AdaBN) [@li2018adabn] is then executed by freezing all learnable network weights and recalibrating the running mean and variance statistics inside the normalization layers using unlabelled target MRI patches.

4.  **Oracle.** The same network evaluated in-domain on source CT data gives the upper-bound oracle: the highest segmentation accuracy this architecture and training budget can reach when no domain shift is present.

> *[Diagram drawn in TikZ — see the LaTeX source.]*

**Figure 1.** Methodology flowchart. A 3D U-Net is trained on CT with Hounsfield unit windowing; its weights are then frozen and target MRI is adapted at inference by percentile intensity normalization followed by Adaptive Batch Normalization, before the three benchmark settings are evaluated.

The three benchmark settings evaluated on the target MRI data are therefore:

- **Naive (lower bound)**: the CT-trained U-Net applied to MRI with no adaptation. It measures the raw cross-modality transfer and exposes the domain gap.

- **Adapted**: the same frozen network with percentile normalization and, cumulatively, AdaBN applied at inference. Because no MRI label is used at any point, this reflects the realistic unsupervised setting in which target annotations are unavailable.

- **Oracle (upper bound)**: the same network evaluated in-domain, which provides the reference against which the recovered fraction of the gap is measured.

Figure 2 illustrates how the settings relate the source and target domains and how the domain gap is measured between the naive lower bound and the oracle.

> *[Diagram drawn in TikZ — see the LaTeX source.]*

**Figure 2.** Experimental design. A single U-Net is trained on labelled CT and evaluated three ways: naively on MRI, on MRI after test-time percentile normalization and AdaBN using only unlabelled target patches, and in-domain as the oracle upper bound. The domain gap closed by adaptation is the improvement over the naive setting as a fraction of the naive-to-oracle interval.

The domain gap closed by adaptation is reported as the improvement of the adapted setting over the naive setting, expressed as a fraction of the naive-to-oracle interval, consistent with the M3DA protocol [@shirokikh2025m3da]. Concretely, if the naive, adapted, and oracle settings achieve scores $s_B$, $s_{DA}$, and $s_O$ respectively, the fraction of the gap closed is $\left(s_{DA} - s_B\right) / \left(s_O - s_B\right)$. A value of zero means the adaptation achieved nothing beyond the naive baseline, whereas a value of one would mean it fully matched the oracle upper bound. This normalised measure is more informative than a raw score difference because it accounts for how large the gap was in the first place.

## 4.2 Hardware and/or Software Components

**Dataset.** The AMOS-22 dataset [@ji2022amos] provides 500 CT and 100 MRI abdominal scans with voxel-level annotations for 15 organs: spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, pancreas, right adrenal gland, left adrenal gland, duodenum, bladder, and prostate/uterus. These organs vary widely in size and shape, from the large and relatively easy liver to the small and thin adrenal glands and esophagus, so the dataset is a demanding test of a model’s ability to segment structures across a wide range of scales. The scans were collected from multiple centres, scanner vendors, imaging phases, and patient conditions, so the dataset captures much of the diversity encountered in real clinical practice. Following the AMOS convention, cases with a numerical identifier at most 500 are CT and those above 500 are MRI, which allows the two modalities to be separated automatically without any manual bookkeeping; the modality is in any case directly detectable from the intensity range. Two of the fifteen classes, the bladder and prostate/uterus, are not annotated in the released MRI subset, so all cross-modality results in this report are reported over the remaining thirteen organs.

**Preprocessing and data pipeline.** Raw abdominal volumes differ substantially in their voxel spacing and overall dimensions, because different scanners and protocols reconstruct images at different resolutions. To place all scans on a common geometric footing, each volume is resampled to a spacing of $1.5 \times 1.5 \times 2.0$ mm using trilinear interpolation for the image and nearest-neighbour interpolation for the label map, the latter being necessary to preserve the discrete integer organ labels. Source CT volumes are then intensity-windowed to the standard soft-tissue Hounsfield range $[-175, 250]$ and rescaled, which is the conventional preprocessing for abdominal CT and exploits the fact that Hounsfield units are physically calibrated and therefore directly comparable between scans. Because abdominal organs occupy only a small fraction of the total image volume, uniform random cropping would produce patches that are almost entirely background; instead, foreground-biased three-dimensional patches of size $96 \times 96 \times 96$ are sampled, so that a substantial proportion of every training batch is centred on organ tissue rather than on empty background.

**Model and training.** The segmentation network is a three-dimensional U-Net [@cicek20163dunet; @he2016resnet] implemented with the MONAI framework [@cardoso2022monai]. It follows the canonical encoder–decoder structure: the encoder repeatedly applies convolutional blocks and downsampling to build an increasingly abstract, low-resolution representation of the input, while the decoder mirrors this with upsampling and convolution to recover full resolution, and skip connections carry fine spatial detail directly from encoder to decoder so that boundaries are not lost. Residual connections within each block ease the optimisation of the deep network [@he2016resnet]. The network has five resolution levels with channel widths of 32, 64, 128, 256, and 320 and downsampling strides of $(2, 2, 2, 2)$, and produces a 16-channel output corresponding to the background and the 15 organ classes.

The network uses batch normalization [@ioffe2015batchnorm] throughout. The choice is deliberate: batch normalization maintains running estimates of the per-channel mean and variance of its inputs, and these buffers are what the AdaBN step re-estimates on the target domain [@li2018adabn]. A backbone built on instance normalization [@ulyanov2016instance], which normalises each sample independently and keeps no running statistics, would leave nothing for AdaBN to recalibrate.

Training minimises a combined Dice and cross-entropy loss [@milletari2016vnet]. The Dice loss directly optimises the overlap between the predicted and ground-truth regions and is well suited to the severe class imbalance of medical images, while the cross-entropy term provides a smooth, well-behaved gradient for every voxel. For a predicted probability map $p$ and a one-hot ground truth $g$ over $C$ classes, the two components can be written as $$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C}\sum_{c=1}^{C}
\frac{2\sum_{v} p_{c,v}\, g_{c,v}}{\sum_{v} p_{c,v} + \sum_{v} g_{c,v}},
\qquad
\mathcal{L}_{\text{CE}} = -\frac{1}{|V|}\sum_{v}\sum_{c=1}^{C}
g_{c,v} \log p_{c,v},$$ where $v$ indexes voxels, and the total loss is their sum, $\mathcal{L} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}$. Optimisation uses the AdamW optimiser [@loshchilov2019adamw], which decouples weight decay from the adaptive gradient update and thereby regularises more reliably than classical Adam, at a learning rate of $2 \times 10^{-4}$ with a batch size of two patches, together with automatic mixed precision that stores most tensors in half precision to roughly halve memory use and accelerate computation on the graphics processing unit. Table 1 lists the full configuration.

| **Parameter**  | **Value**                 | **Role**                                          |
|:---------------|:--------------------------|:--------------------------------------------------|
| `SEED`         | 42                        | Fixed random seed for reproducibility             |
| `CT_MAX_ID`    | 500                       | Case identifiers $\leq 500$ are CT, above are MRI |
| `NUM_CLASSES`  | 16                        | 15 organs plus background                         |
| `SPACING`      | $(1.5, 1.5, 2.0)$ mm      | Common resampled voxel spacing                    |
| `CT_WINDOW`    | $(-175, 250)$ HU          | Soft-tissue window for source CT                  |
| `PATCH`        | $(96, 96, 96)$            | Foreground-biased training patch size             |
| `CHANNELS`     | $(32, 64, 128, 256, 320)$ | U-Net channel widths per level                    |
| `STRIDES`      | $(2, 2, 2, 2)$            | Downsampling strides                              |
| `BATCH`        | 2                         | Patches per optimisation step                     |
| `LR`           | $2 \times 10^{-4}$        | AdamW learning rate                               |
| `NUM_EPOCHS`   | 300                       | Source training length                            |
| `VAL_INTERVAL` | 5                         | Epochs between validation passes                  |
| `CACHE_RATE`   | 0.0                       | No in-memory caching, to bound host RAM use       |

**Table 1.** Hyper-parameter configuration and data preprocessing parameters.

**The adaptation method.** The domain adaptation itself consists of two steps, both applied at inference and neither requiring a target label, a gradient update, or a second network.

*Step 1: percentile intensity normalization.* MRI intensities are arbitrary and scanner-dependent, so the fixed Hounsfield window that is appropriate for CT maps most MRI voxels to a meaningless clipped value. Instead, each MRI volume is rescaled by its own $0.5$ and $99.5$ intensity percentiles into the $[0,1]$ range the network expects. Writing $q_{0.5}(x)$ and $q_{99.5}(x)$ for those percentiles of volume $x$, the normalised volume is $$\hat{x} \;=\; \operatorname{clip}\!\left(
\frac{x - q_{0.5}(x)}{q_{99.5}(x) - q_{0.5}(x)},\; 0,\; 1 \right).$$ Using percentiles rather than the raw minimum and maximum makes the mapping robust to the few extreme outlier voxels that are common in MRI, and the same principle underlies the intensity normalization used by nnU-Net [@isensee2021nnunet]. The step is per-volume, label-free, and costs a few lines of code.

*Step 2: Adaptive Batch Normalization.* Percentile scaling fixes the input representation, but the internal feature distributions of the network are still calibrated to CT. AdaBN [@li2018adabn] addresses this by freezing every learnable weight and bias in the network and passing unlabelled target MRI patches forward so that the running mean $\mu$ and variance $\sigma^2$ buffers of each batch normalization layer are re-estimated from target data. No gradient is computed and no parameter is learned; only the normalization statistics change. Because these statistics are exactly the quantities that summarise the distribution of activations, replacing the source estimates with target estimates re-centres and re-scales the features at every depth of the network, aligning the feature distributions across the two modalities.

Because neither step requires target labels, source data, or a second network, the method is far cheaper than adversarial image-translation approaches [@zhu2017cyclegan; @chen2020sifa], which is what makes it practical on a single consumer graphics processing unit and applicable to a model that has already been deployed. Table 2 lists the principal software and hardware tools.

| **Tool**                    | **Function**             | **Why selected**                                          |
|:----------------------------|:-------------------------|:----------------------------------------------------------|
| Python 3.12                 | Programming language     | Standard for deep learning research                       |
| PyTorch                     | Deep learning framework  | GPU-accelerated tensor computation                        |
| MONAI [@cardoso2022monai] | Medical imaging toolkit  | Ready 3D U-Net, losses, metrics, sliding-window inference |
| NiBabel / NumPy             | Volume I/O and arrays    | Reading and resampling NIfTI scans                        |
| NVIDIA RTX 3090             | Graphics processing unit | Consumer-grade 3D training                                |

**Table 2.** Software and hardware tools used in the project.

## 4.3 Hardware and/or Software Implementation

The pipeline is implemented in cooperating scripts, each with a single clear responsibility, so that the individual stages can be run, debugged, and re-run independently. A preprocessing script reads the selected AMOS-22 volumes from disk, resamples each to the common $1.5 \times 1.5 \times 2.0$ mm spacing, and writes the results together with their label maps; this stage is executed once and its output is reused by every subsequent experiment, which avoids repeatedly paying the cost of reading and resampling the large native volumes. In-memory caching is disabled (`CACHE_RATE` $= 0.0$) so that the working set is streamed from disk and the host memory footprint stays within the 16 GB available.

A training script loads the cases, separates them into the CT source and MRI target partitions by their identifiers, and trains the source U-Net on CT for 300 epochs. Validation is performed every fifth epoch on a held-out split, and the checkpoint achieving the best validation mean Dice is retained rather than the checkpoint from the final epoch, which guards against over-training. All random seeds are fixed at 42 and cuDNN benchmarking is enabled, so that runs are reproducible up to the non-determinism inherent in benchmarked convolution kernels.

A separate adaptation and evaluation script loads the frozen source checkpoint and carries out the three benchmark evaluations. For the naive setting it applies the network directly to MRI volumes preprocessed exactly as CT was. For the adapted setting it re-normalises each MRI volume by its own $0.5$–$99.5$ percentiles and, for the AdaBN variant, first performs a statistics-only forward pass over unlabelled target patches to refresh the batch normalization buffers before evaluation. For the oracle setting it evaluates the same checkpoint in-domain on held-out CT. In every case, full volumes are segmented by sliding-window inference, in which the network is applied to overlapping $96^3$ patches that tile the volume and whose predictions are aggregated into a single dense label map. Because the adaptation is confined to inference, the naive, percentile, and AdaBN results are all produced from one and the same set of trained weights, so any difference between them is attributable to the adaptation alone and not to training variance. All experiments were run locally with automatic mixed precision so as to fit comfortably within the memory available on the graphics processing unit.

# Chapter 5: Investigation/Experiment, Result, Analysis and Discussion

This chapter reports the experiments that quantify the CT-to-MRI domain gap for abdominal multi-organ segmentation and the extent to which test-time adaptation recovers it. A three-dimensional U-Net was trained on Computed Tomography (CT) and then evaluated both *in-domain* on held-out CT and *out-of-domain* on Magnetic Resonance Imaging (MRI), without any MRI supervision. This source-only setting is the standard baseline against which any domain adaptation method must be measured: the difference between the in-domain and out-of-domain scores is the domain gap. Performance is reported with the Dice similarity coefficient, the standard overlap metric for volumetric segmentation [@taha2015metrics], averaged over the thirteen organs annotated in the AMOS-22 MRI subset.

## 5.1 The Cross-Modality Domain Gap

The intensity statistics of the two modalities barely overlap. CT values are expressed in calibrated Hounsfield units and cluster near $-1000$ for air, whereas MRI intensities are arbitrary and spread broadly around zero with no fixed physical meaning. Because the network learns to associate organs with CT-specific intensity signatures that do not exist in MRI, a large performance drop is expected when the model is applied across modalities.

Figure 3 shows the effect during training. In-domain CT validation Dice rises quickly and then plateaus in the low $0.8$ range, while the same checkpoints evaluated on unadapted MRI stall around roughly $0.40$–$0.50$ and never improve further, no matter how long source training continues. The two curves separate within the first fifty epochs and never converge. The gap is therefore not a symptom of an undertrained network, since more source training does nothing to close it; it follows from the mismatch between the two intensity distributions.

Quantitatively, the naive cross-modality transfer reaches a mean Dice of only $0.495$ on MRI, against an in-domain oracle of approximately $0.80$ for the same architecture. The model therefore loses more than half of its accuracy when the scanner modality changes.

![Figure 3](figures/training_curves.png)

**Figure 3.** Validation mean Dice during source training, comparing in-domain CT performance against unadapted out-of-domain MRI evaluation. The CT curve plateaus in the low $0.8$ range while the unadapted MRI curve stalls near $0.4$–$0.5$, a degradation of over 50% caused by domain shift alone.

## 5.2 Percentile Normalization

The domain gap documented above is, to a large extent, an *intensity-representation* problem: MRI volumes have arbitrary, scanner-dependent intensity ranges, so the fixed Hounsfield-unit window used for CT maps most MRI voxels to a meaningless clipped value. This motivates the first adaptation step, in which each MRI volume is rescaled by its own $0.5$–$99.5$ intensity percentiles into the $[0,1]$ range the network expects. The step is applied only at inference and requires no retraining and no target labels; the model is unchanged and only the input representation is standardised.

Mean MRI Dice rises from $0.495$ to $0.642$, a gain of $+0.147$ Dice, or roughly a $30\%$ relative improvement, at no training cost. The improvement is concentrated where the domain gap was largest. The large, high-contrast organs that already transferred reasonably well change little: the aorta is essentially unchanged ($0.630 \to 0.628$) and the spleen gains $+0.046$. The middle- and low-tier organs gain far more, with the gallbladder more than doubling ($0.173 \to 0.410$), the left adrenal gland rising from $0.343$ to $0.574$, and the stomach and pancreas rising by $+0.197$ and $+0.145$ respectively. Normalization therefore acts on the organs whose CT and MRI appearance differ most.

## 5.3 Adaptive Batch Normalization

Percentile scaling repairs the input representation, but the internal feature statistics of the network remain calibrated to CT. The second adaptation step, AdaBN [@li2018adabn], freezes all learned weights and recalibrates only the running mean and variance buffers of the batch normalization layers using unlabelled target MRI patches.

Stacked on top of percentile normalization, AdaBN raises mean MRI Dice from $0.642$ to $0.721$, a further gain of $+0.079$. The recovery is consistent across organs: twelve of the thirteen improve, with the left kidney the single exception, dipping slightly from $0.782$ to $0.748$. Because the left kidney was already close to its own oracle ceiling of $0.769$, this dip is small in absolute terms and does not affect the overall trend.

Table 3 summarises the full adaptation ladder, and Table 4 gives the organ-level breakdown that Figure 4 visualises.

| **Stage** | **Setting**                    |   **Mean Dice**    | **Labels?** | **Gap closed** |
|:----------|:-------------------------------|:------------------:|:-----------:|:--------------:|
| 1         | Naive transfer                 |       0.495        |     No      |       —        |
| 2         | Percentile normalization       |       0.642        |     No      |     48.2%      |
| 3         | Percentile $+$ AdaBN           |     **0.721**      |     No      |   **74.1**%    |
| 4         | Oracle (in-domain upper bound) | ~0.80 |     N/A     |       —        |

**Table 3.** The adaptation ladder. Mean MRI Dice at each stage, with the fraction of the naive-to-oracle domain gap recovered. No stage uses target labels or updates any learned weight.

| **Organ**           | **Naive** | **Percentile** | **$+$AdaBN** | **Oracle** | **Total gain** |
|:--------------------|:---------:|:--------------:|:------------:|:----------:|:--------------:|
| Spleen              |   0.750   |     0.796      |    0.855     |   0.947    |    $+0.105$    |
| Right kidney        |   0.707   |     0.765      |    0.825     |   0.936    |    $+0.118$    |
| Left kidney         |   0.727   |     0.782      |    0.748     |   0.769    |    $+0.021$    |
| Gallbladder         |   0.173   |     0.410      |    0.468     |   0.679    |    $+0.295$    |
| Esophagus           |   0.391   |     0.437      |    0.496     |   0.716    |    $+0.105$    |
| Liver               |   0.769   |     0.861      |    0.920     |   0.960    |    $+0.151$    |
| Stomach             |   0.533   |     0.730      |    0.789     |   0.904    |    $+0.256$    |
| Aorta               |   0.630   |     0.628      |    0.687     |   0.907    |    $+0.057$    |
| Postcava (IVC)      |   0.483   |     0.499      |    0.558     |   0.849    |    $+0.075$    |
| Pancreas            |   0.589   |     0.734      |    0.793     |   0.834    |    $+0.204$    |
| Right adrenal gland |   0.266   |     0.360      |    0.419     |   0.670    |    $+0.153$    |
| Left adrenal gland  |   0.343   |     0.574      |    0.586     |   0.607    |    $+0.243$    |
| Duodenum            |   0.321   |     0.445      |    0.504     |   0.741    |    $+0.183$    |

**Table 4.** Per-organ MRI Dice at each stage of the adaptation ladder, over the thirteen organs annotated in the AMOS-22 MRI subset. The rightmost column is the total gain of the fully adapted setting over the naive baseline.

![Figure 4](figures/per_organ_recovery.jpg)

**Figure 4.** Per-organ performance recovery via unsupervised domain adaptation. Per-organ Dice comparing the unadapted baseline, intensity alignment by percentile normalization, feature-distribution alignment by AdaBN, and the oracle upper bound. Combining percentile normalization with AdaBN yields incremental gains on nearly every organ, progressively closing the domain gap toward the oracle.

## 5.4 Ablation Study

To confirm that both adaptation components contribute, each was evaluated in isolation and in combination on the same frozen source checkpoint. Table 5 reports the result. Because no weight is updated in any configuration, every row is produced from identical trained parameters, so the differences are attributable to the adaptation alone.

| **Percentile norm.** | **AdaBN** | **Mean MRI Dice** |
|:--------------------:|:---------:|:-----------------:|
|          –           |     –     |       0.495       |
|          –           |           |       \[–\]       |
|                      |     –     |       0.642       |
|                      |           |     **0.721**     |

**Table 5.** Ablation of the two adaptation components. All configurations use the same frozen CT-trained checkpoint and no target labels.

## 5.5 Qualitative Results

Figure 5 shows a representative MRI case with the ground-truth mask, the unadapted prediction, and the prediction after adaptation. The naive baseline exhibits severe prediction loss: organ boundaries dissolve and several structures are missed entirely, which is what a mean Dice near $0.50$ looks like in practice. After intensity and statistical alignment, organ boundaries are visibly restored and the large organs are recovered in approximately the correct locations and shapes, although the result remains rougher than the reference mask. The qualitative picture therefore matches the quantitative one: adaptation recovers much of the lost structure without closing the gap entirely.

![Figure 5](figures/gt_naive_normalization_comparison.png)

**Figure 5.** Visual comparison of target MRI segmentation outputs against ground-truth masks, showing severe prediction loss in the unadapted baseline versus restored organ boundaries following intensity normalization and AdaBN.

## 5.6 Further Exploration: CycleGAN Image Translation

In addition to the test-time adaptation reported above, a two-dimensional CycleGAN [@zhu2017cyclegan] was trained to translate CT slices into synthetic MRI, with the intention of training a segmenter on the translated images and thus obtaining a model that has effectively seen MRI-like data without ever seeing a real MRI label.

The translation component worked. The generated images were anatomically faithful: the body outline, the spine, and the positions of the major organs were preserved, and the characteristic CT-to-MRI contrast inversion was reproduced correctly. The segmentation component did not. At the data scale available for this project, the two-dimensional segmenter trained on translated slices under-fit and failed to beat the naive baseline, so the approach was not pursued further. The bottleneck was the segmenter and the quantity of two-dimensional training slices rather than the translation itself, which suggests that the natural continuation is a three-dimensional CycleGAN paired with a stronger segmentation backbone and a larger training set. This negative result is reported here because it directly shaped the decision to concentrate on lightweight, inference-time adaptation, which delivered a measurable gain at a small fraction of the computational cost.

## 5.7 Discussion

Five conclusions follow from these experiments.

First, the source model is strong in-domain, reaching a mean Dice of approximately $0.80$ on CT, so the low naive MRI scores reflect domain shift rather than an undertrained network. The baseline is a fair, well-optimised reference, and Figure 3 confirms that the gap does not narrow with longer training.

Second, the CT-to-MRI gap is large but not total. Mean Dice roughly halves, and the model retains real skill on large organs such as the liver and spleen while failing badly on small ones such as the gallbladder and the adrenal glands. This difficulty ordering matches the pattern reported in the original AMOS-22 benchmark [@ji2022amos] and in the broader literature [@guan2021survey; @shirokikh2025m3da].

Third, a substantial part of that gap comes from the intensity representation rather than from the anatomy. Label-free percentile harmonisation alone recovers $+0.147$ mean Dice with no retraining, and the gain is concentrated on the organs that needed it most. This is a practical, resource-efficient result: a meaningful improvement obtained by changing a few lines of preprocessing, requiring neither target labels nor a second network, in contrast to heavier adversarial methods [@zhu2017cyclegan; @chen2020sifa].

Fourth, feature-statistic alignment adds a further, smaller, but consistent increment. AdaBN contributes $+0.079$ on top of percentile normalization, taking the total recovery to $74\%$ of the naive-to-oracle interval. That the two steps compose so cleanly supports the interpretation that they correct different things: percentile scaling fixes the input distribution, AdaBN fixes the internal activation distributions that remain misaligned afterwards. The recovered fraction is measured against an oracle defined by the same architecture and training budget, so it states how much of *this* model’s lost accuracy is recoverable and is not a claim of state-of-the-art absolute accuracy.

Fifth, the approach has a clear ceiling. A residual gap of roughly $0.08$ mean Dice to the oracle remains, and it is largest for the small, thin, low-contrast structures: the right adrenal gland reaches only $0.419$ against an oracle of $0.670$, and the postcava reaches $0.558$ against $0.849$. Statistical alignment cannot recover organs whose failure is structural rather than purely intensity-driven. Closing that residual gap would require a learned adaptation method that models anatomy and not only intensity distributions, discussed further in Section 9.3.

# Chapter 6: Impacts of the Project

## 6.1 Impact on Societal, Health, Safety, Legal and Cultural Issues

Automated abdominal organ segmentation has direct health and societal value. It can accelerate radiotherapy planning, support computer-aided diagnosis, and reduce the manual workload of radiologists, which may improve access to quality care where expert time is scarce. By studying cross-modality adaptation, this project contributes to making such tools robust across the different scanners and modalities found in real hospitals, which is essential for safe clinical use. Because the adaptation used here operates entirely at inference time, it can be applied to a model that is already installed in a clinical system without retraining it or returning to the original training archive. A domain-shifted model can produce silently incorrect segmentations, so the project also points to a safety requirement: models must be validated on the target modality before deployment, and the residual gap reported in Chapter 5 shows that adaptation reduces but does not eliminate the need for that validation. From a legal and ethical standpoint, the project uses only the publicly released, de-identified AMOS-22 dataset under its licence, respecting patient privacy and data-governance requirements.

## 6.2 Impact on Environment and Sustainability

Computational efficiency was a deliberate design choice. By resampling the data to a common spacing, using mixed-precision training, and adopting an adaptation method that requires no retraining and no second network, the entire pipeline runs on a single consumer graphics processing unit rather than a large data centre. The adaptation step itself costs only a forward pass over unlabelled target patches, which is negligible next to the cost of training an adversarial translation model. That reduces the energy consumption and carbon footprint of the work, and it lowers the barrier for researchers with limited resources. Efficient adaptation also reduces the need to collect and annotate large new datasets for every modality, saving both human effort and the resources involved in additional data acquisition.

# Chapter 7: Project Planning and Budget

The project was carried out over one academic term. The planning followed four phases: (1) background study and literature review; (2) dataset preparation and pipeline implementation; (3) experiments, adaptation, and evaluation; and (4) analysis and report writing. A Gantt chart summarising the schedule is shown in Figure 6, and an indicative budget is given in Table 6.

![Figure 6](figures/gchart.png)

**Figure 6.** Project schedule Gantt chart.

| **Item**           |     **Description**      | **Approx. Cost (BDT)** |
|:-------------------|:------------------------:|:----------------------:|
| Computing hardware | Consumer GPU workstation | Provided by University |
| Storage            | Dataset and checkpoints  | Provided by University |
| Miscellaneous      | Printing, documentation  |          1200          |
| **Total**          |                          |          1200          |

**Table 6.** Indicative project budget.

Because the dataset is publicly available and all software is open-source, the principal costs of this project were the computing hardware and the associated electricity, both of which were kept low by the resource-efficient design.

# Chapter 8: Complex Engineering Problems and Activities

## 8.1 Complex Engineering Problems (CEP)

Table 7 maps the complex engineering problem attributes to this project.

|     | **Attribute**              | **Addressing the problem in the project**                                                                           |
|:----|:---------------------------|:--------------------------------------------------------------------------------------------------------------------|
| P1  | Depth of knowledge (K3–K8) | Requires machine learning, medical image analysis, 3D deep learning, optimisation, and research literature (WK8).   |
| P2  | Conflicting requirements   | Trade-off between model accuracy, GPU memory, and training time on limited hardware.                                |
| P3  | Depth of analysis          | No unique solution; choice among architectures, normalization schemes, and adaptation strategies requires analysis. |
| P4  | Familiarity of issues      | 3D U-Net, domain shift, test-time adaptation, and evaluation metrics.                                               |
| P5  | Applicable codes/standards | No fixed standard; evaluation follows established segmentation metric conventions.                                  |
| P6  | Stakeholder involvement    | Clinicians, patients, and researchers as end-users of segmentation tools.                                           |
| P7  | Interdependence            | Interdependent sub-systems: preprocessing, data loading, model, training, adaptation, and evaluation.               |

**Table 7.** Complex engineering problem attributes.

## 8.2 Complex Engineering Activities (CEA)

Table 8 maps the complex engineering activity attributes to this project.

|     | **Attribute**                       | **Addressing the activity in the project**                                                                                                   |
|:----|:------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| A1  | Range of resources                  | Uses human effort, a public dataset, open-source software, and a consumer GPU.                                                               |
| A2  | Level of interactions               | Coordination among preprocessing, training, adaptation, and evaluation components, and with the supervisor.                                  |
| A3  | Innovation                          | Applies a lightweight, label-free, inference-time adaptation method to a public cross-modality benchmark.                                    |
| A4  | Consequences to society/environment | Supports healthcare while minimising computational and energy costs.                                                                         |
| A5  | Familiarity                         | Requires familiarity with deep learning frameworks, medical imaging formats, and evaluation metrics. UN SDG \#3: Good Health and Well-being. |

**Table 8.** Complex engineering activity attributes.

# Chapter 9: Conclusions

## 9.1 Summary

This project investigated cross-modality domain shift between CT and MRI for abdominal multi-organ segmentation on the AMOS-22 dataset. A three-dimensional U-Net was trained on CT with standard Hounsfield unit windowing and evaluated on MRI to quantify the domain gap, and a two-step unsupervised adaptation combining dynamic $0.5$–$99.5$ percentile intensity normalization with Adaptive Batch Normalization was applied at inference time to reduce it. In-domain CT performance provided the oracle upper bound. The results show that the severe performance loss caused by CT-to-MRI domain shift can be reduced substantially without retraining the model: naive transfer achieves a mean MRI Dice of $0.495$, percentile normalization raises this to $0.642$, and adding AdaBN raises it further to $0.721$ against an oracle of approximately $0.80$, recovering about $74\%$ of the naive-to-oracle interval. Aligning the intensity distributions first and the feature distributions second produced consistent, stepwise organ-wise Dice recoveries toward the upper bound. The entire pipeline was made resource-efficient enough to run on a single consumer graphics processing unit, and the finding that a partial but meaningful recovery is achievable at negligible cost matches the conclusions of the M3DA benchmark [@shirokikh2025m3da].

## 9.2 Limitations

The project has several limitations. Owing to hardware constraints, the experiments were run on a resampled version of the dataset with a fixed hyper-parameter configuration rather than a systematic search, and the MRI evaluation set is small, which makes per-organ metrics for the smallest structures noisy. Two of the fifteen AMOS-22 classes could not be assessed across modalities at all, because the bladder and prostate/uterus are not annotated in the released MRI subset. Performance is reported using the Dice coefficient alone; boundary-based measures such as the 95th-percentile Hausdorff and average symmetric surface distances [@taha2015metrics] were not computed, so the report cannot speak to whether the recovered segmentations are also boundary-accurate, which matters clinically. The adaptation method is intentionally lightweight and does not include adversarial image translation or learned feature alignment, which can achieve higher accuracy at greater computational cost; the CycleGAN experiment described in Section 5.6 was attempted but under-fit at the available data scale. Finally, training was limited in scale relative to the full-length schedules used in the original benchmarks, so the absolute accuracy values are below the state of the art even though the relative trends are faithful.

## 9.3 Future Improvement

Test-time intensity and statistical alignment give an immediate and inexpensive recovery, but future work will focus on unsupervised domain adaptation frameworks that learn from the target distribution rather than only re-estimating statistics over it. Four directions follow from the results reported here.

First, and most directly, the residual gap is concentrated in small, thin, and low-contrast structures whose failure appears to be structural rather than purely intensity-driven. Graph Neural Networks are a promising way to address this, because they can model cross-modality structural topology and inter-organ spatial relationships explicitly, supplying the anatomical prior that a purely intensity-based correction cannot. Second, the CycleGAN translation [@zhu2017cyclegan] should be scaled to three dimensions and paired with a stronger segmentation backbone; the translations themselves were anatomically faithful, so the bottleneck lay in the two-dimensional segmenter rather than in the translation. Third, training on the full AMOS-22 dataset, combined with histogram matching and bias-field correction, would address the residual acquisition-level variation that percentile scaling leaves untouched. Fourth, established adaptation families that were outside the scope of this project deserve direct comparison on the same benchmark, including SIFA [@chen2020sifa], entropy minimisation [@vu2019advent], Fourier domain adaptation [@yang2020fda], and gradient-based test-time adaptation [@karani2021test], alongside transformer backbones [@tang2022swinunetr] and the additional shifts catalogued in the M3DA benchmark [@shirokikh2025m3da]. Extending the evaluation to boundary-aware metrics [@taha2015metrics] would in each case give a fuller picture than Dice alone.
