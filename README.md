<h1 align="center"> SB-Bench: <u>S</u>tereotype <u>B</u>ias <u>Bench</u>mark for Large Multimodal Models</h1>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="SB-Bench">
</p>

<p align="center">
   <a href="https://github.com/UCF-CRCV/SB-Bench/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-CC--BY--NC%204.0-blue" alt="license"></a>
</p>

[Vishal Narnaware](https://www.linkedin.com/in/vishalnarnaware/)* , [Ashmal Vayani](https://scholar.google.com/citations?user=K4KF1SwAAAAJ&hl=en&oi=ao)* , [Rohit Gupta](https://scholar.google.com/citations?user=0WukQpMAAAAJ&hl=en&oi=ao)<sup>♠</sup> , [Swetha Sirnam](https://scholar.google.com/citations?user=XwocaTcAAAAJ&hl=en&oi=ao)<sup>♠</sup> , [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en&oi=ao)
###### * Equal Authors
#### Center for Research in Computer Vision (University of Central Florida), Amazon

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Access-<COLOR>)]()
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)]()

#### Official GitHub repository for  `SB-Bench: Stereotype Bias Benchmark for Large Multimodal Models`.
---

## 📢 Latest Updates
- **Feb-05-25**- Technical report of SB-Bench is released on [arxiv]()! 🔥🔥
- **Feb-05-25**- *SB-Bench Dataset* and codes are released. It provides 7,500 visually grounded, non-synthetic multiple-choice QA samples across 9 social bias categories to extensively evaluate the performance of LMMs. 🔥🔥

---

## 🏆 Highlights

![main figure](Assets/teaser_merged.jpg)
> <p align="justify"> <b> <span style="color: blue;">Figure</span></b>: <i>(Left): </i> The image presents a scenario where a family is selecting a babysitter between a university student and a transgender individual. Notably, all LMMs exhibit bias by consistently favoring the university student as the more trustworthy choice. These responses highlight how LMMs reinforce societal stereotypes, underscoring the need for improved bias evaluation and mitigation strategies. <i>(Right): </i> The SB-Bench includes nine diverse domains and 60 sub-domains to rigorously assess the performance of LMMs in visually grounded stereotypical scenarios. SB-Bench comprises over 7.5k questions on carefully curated non-synthetic images.

> **<p align="justify"> Abstract:** Stereotype biases in Large Multimodal Models (LMMs) perpetuate harmful societal prejudices, undermining the fairness and equity of AI applications. As LMMs grow increasingly influential, addressing and mitigating inherent biases related to stereotypes, harmful generations, and ambiguous assumptions in real-world scenarios has become essential. However, existing datasets evaluating stereotype biases in LMMs often lack diversity and rely on synthetic images, leaving a gap in bias evaluation for real-world visual contexts. To address the gap in bias evaluation using real images, we introduce the Stereotype Bias Benchmark (`SB-Bench`), the most comprehensive framework to date for assessing stereotype biases across nine diverse categories with non-synthetic images. `SB-Bench` rigorously evaluates LMMs through carefully curated, visually grounded scenarios, challenging them to reason accurately about visual stereotypes. It offers a robust evaluation framework featuring real-world visual samples, image variations, and multiple-choice question formats. By introducing visually grounded queries that isolate visual biases from textual ones, `SB-Bench` enables a precise and nuanced assessment of a model’s reasoning capabilities across varying levels of difficulty. Through rigorous testing of state-of-the-art open-source and closed-source LMMs, `SB-Bench` provides a systematic approach to assessing stereotype biases in LMMs across key social dimensions. This benchmark represents a significant step toward fostering fairness in AI systems and reducing harmful biases, laying the groundwork for more equitable and socially responsible LMMs. Our code and dataset are publically available. </p>

## `SB-Bench` provides a more rigorous and standardized evaluation framework for next-generation multilingual LMMs.

**Main contributions:**
1) **`Stereotype-Bias Benchmark (SB-Bench):`** We introduce SB-bench, a diverse multiple-choice benchmark featuring 7,500 non-synthetic visual samples that span across nine categories and 60 subcategories of social biases, providing a more accurate reflection of real-world contexts.
2) **`Visually Grounded Scenarios:`** SB-bench is meticulously designed to introduce visually grounded scenarios, explicitly separating visual biases from textual biases. This enables a focused and precise evaluation of visual stereotypes in LMMs.
3) **`Comprehensive Evaluation:`** We benchmark both open-source and closed-source LMMs, along with their various scale variants, on SB-bench. Our analysis highlights critical challenges and provides actionable insights for developing more equitable and fair multimodal models.
   
<hr />

## 🗂️ Dataset

<p align="center">
   <img src="Assets/dataset_compare.png" alt="Dataset Comparison table"></a>
</p>

> <p align="justify"> <b> <span style="color: blue;">Table</span></b>: Comparison of various LMM evaluation benchmarks with a focus on stereotype bias. Our approach is one of only three to assess nine bias types, is based on real images, unlike B-AVIBench, and unlike the Open-Ended BiasDora is easy to evaluate because of its Multiple-Choice design. The <i>Question Types</i> are classified as ‘ITM‘ (Image-Text Matching), ‘OE’ (Open-Ended) or MCQ (Multiple-Choice). </p>

#### `SB-Bench` comprises of nine social bias categories.
<p align="center">
   <img src="Assets/dataset_describe.png" alt="Dataset Comparison table"></a>
</p>

> <p align="justify"> <b> <span style="color: blue;">Table</span></b>: Bias Types: Examples from the nine bias categories. The source which identifies the bias is reported. </p>

#### Below we present the dataset distribution in `SB-Bench` across nine social bias categories.
<p align="center">
   <img src="Assets/category_distribution_v2.png" alt="Category Comparison table"></a>
</p>

<hr />

## 🔍 Dataset Annotation Process
![main figure](Assets/multimodal_bias_pipeline.jpg)
> <p align="justify"> <b> <span style="color: blue;">Figure</span></b>: `SB-Bench` pipeline: We start with text bias evaluation question for a stereotype which includes descriptive text context detailing the scene and bias probing question. A visual query generator then transforms this context into a search-friendly query, retrieving real-world images from the web. The retrieved images are filtered using CLIP to ensure relevance. The visual information remover anonymizes text references to prevent explicit leakage. The text is paired with selected visual content along with the bias probing question to create the multi-modal bias evaluation benchmark. </p>


<p align="center">
   <img src="Assets/paired_image_pipeline.jpg" width=500 alt="Paired Image Pipeline"></a>
</p>

> <p align="justify"> <b> <span style="color: blue;">Figure</span></b>: Paired Images Pipeline: For dual-image queries, the Dual Query Generator creates two separate queries, each independently sent to a web search. We then retrieve the top 5 images per query and generate 25 paired combinations by stitching images side by side. This approach differentiates dual-image queries from single-image queries, ensuring diverse pairwise representations. </p>


<hr />

## 📊 Results
The below Barplot presents the Bias-Score evaluation results of various recent LMMs (open-sourced and closed-sourced) across nine diverse stereotype categories.

<p align="center">
   <img src="Assets/main_results.jpg" alt="Main Results"></a>
</p>

![main figure](Assets/results.png)
> <p align="justify"> <b> <span style="color: blue;">Table</span></b>: Evaluation of various LMMs on the `SB-Bench` Benchmark shows that proprietary models like GPT-4o and Gemini-1.5-Flash outperform open-source models in fairness scores. Among open-source models, InternVL2 stands out as the best for generating non-stereotypical responses. </p>

![main figure](Assets/success_failure.jpg)
> <p align="justify"> <b> <span style="color: blue;">Figure</span></b>: We present qualitative examples from both an open-source model, InternVL-2-8B, and a closed-source model, GPT-4o-mini, showcasing their performance on success cases <i>(first row)</i> and failure cases <i>(second row)</i> across various stereotype categories in our `SB-Bench`. For failure cases, we also provide the reasoning generated by both LMMs to explain its output. For instance, The Socio-Economic-Situation category mentions two common professions, chemical engineer and the line cook, and asks the question about who performed better in school. The Descriptive Text Context (context) is ambiguous and does not point to any attested bias. However, the model’s implicit bias on chemical engineer’s professions requiring a profound academic foundation as compared to the cook’s profession generates a biased answer. These examples highlight the model’s gap in generating unbiased answers in ambiguous scenarios and its limitations from not utilizing its inherent bias is noteworthy to the observer here. </p>

<hr />

## 🚀 Getting started with SB-Bench

<hr />

## 📂 License
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. The images in ALM-Bench dataset are collected from public domains and sources (refer to [main paper](https://arxiv.org/abs/2411.16508) for more details) and are for academic research use only. 
By using ALM-Bench, you agree not to use the dataset for any harm or unfair discrimination. Please note that the data in this dataset may be subject to other agreements. Video copyrights belong to the original dataset providers, video creators, or platforms.

## 📜 Citation
If you find our work and this repository useful, please consider giving our repo a star and citing our paper as follows:
```bibtex
asdasd
```
