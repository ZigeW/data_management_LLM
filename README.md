# Data Management for Training LLM

A curated list of training data management for large language model resources. The papers are organized according to our survey paper [Data Management For Training Large Language Models: A Survey](https://arxiv.org/abs/2312.01700).

## Contents

- [Pretraining](#pretraining)
  - [Domain Composition](#domain-composition)
  - [Data Quantity](#data-quantity)
  - [Data Quality](#data-quality)
  - [Relations Among Different Aspects](#relations-among-different-aspects)

- [Supervised Fine-Tuning](#supervised-fine-tuning)
  - [Task Composition](#task-composition)
  - [Data Quality](#data-quality-1)
  - [Data Quantity](#data-quantity-1)
  - [Dynamic Data-Efficient Learning](#dynamic-data-efficient-learning)
  - [Relations Among Different Aspects](#relations-among-different-aspects-1)

- [Useful Resources](#useful-resources)

## Pretraining

### Domain Composition

- Lamda: Language models for dialog applications (Arxiv, Jan. 2022) [[Paper]](https://arxiv.org/abs/2201.08239) [[Code]](https://github.com/conceptofmind/LaMDA-rlhf-pytorch)
- Data Selection for Language Models via Importance Resampling (Arxiv, Feb. 2023) [[Paper]](https://arxiv.org/abs/2302.03169) [[Code]](https://github.com/p-lambda/dsir)
- CodeGen2: Lessons for Training LLMs on Programming and Natural Languages (ICLR 2023) [[Paper]](https://arxiv.org/abs/2305.02309) [[Model]](https://github.com/salesforce/CodeGen2)
- DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.10429) [[Code]](https://github.com/sangmichaelxie/doremi)
- A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.13169)
- SlimPajama-DC: Understanding Data Combinations for LLM Training (Arxiv, Sep. 2023) [[Paper]](https://arxiv.org/abs/2309.10818) [[Model]](https://huggingface.co/MBZUAI-LLM) [[Dataset]](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- DoGE: Domain Reweighting with Generalization Estimation (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.15393) [[Code]](https://github.com/Olivia-fsm/DoGE)
- Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance (Arxiv, Mar. 2024) [[Paper]](https://arxiv.org/abs/2403.16952) [[Code]](https://github.com/yegcjs/mixinglaws)
- Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning (ICLR 2024) [[Paper]](https://arxiv.org/abs/2310.06694) [[Code]](https://github.com/princeton-nlp/LLM-Shearing)

### Data Quantity

- #### Scaling Laws
  
  - Scaling Laws for Neural Language Models (Arxiv, Jan. 2020) [[Paper]](https://arxiv.org/abs/2001.08361)
  - An empirical analysis of compute-optimal large language model training (NeurIPS 2022) [[Paper]](https://papers.nips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html)
  - Unraveling the Mystery of Scaling Laws: Part I (Arxiv, Mar. 2024) [[Paper]](https://arxiv.org/abs/2403.06563)

- #### Data Repetition
  
  - Scaling Laws and Interpretability of Learning from Repeated Data (Arxiv, May 2022) [[Paper]](https://arxiv.org/abs/2205.10487)
  - Will we run out of data? An analysis of the limits of scaling datasets in Machine Learning (Arxiv, Oct. 2022) [[Paper]](https://arxiv.org/abs/2211.04325)
  - Scaling Data-Constrained Language Models (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.16264) [[Code]](https://github.com/huggingface/datablations)
  - To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.13230)
  - D4: Improving LLM Pretraining via Document De-Duplication and Diversification (Arxiv, Aug. 2023) [[Paper]](https://arxiv.org/abs/2308.12284)

### Data Quality

- #### Quality Filtering

  - An Empirical Exploration in Quality Filtering of Text Data (Arxiv, Sep. 2021) [[Paper]](https://arxiv.org/abs/2109.00698) 
  - Quality at a glance: An audit of web-crawled multilingual datasets (ACL 2022) [[Paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00447/109285/Quality-at-a-Glance-An-Audit-of-Web-Crawled)
  - The MiniPile Challenge for Data-Efficient Language Models (Arxiv, April 2023) [[Paper]](https://arxiv.org/abs/2304.08442) [[Dataset]](https://huggingface.co/datasets/JeanKaddour/minipile)
  - A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.13169)
  - Textbooks Are All You Need (Arxiv, Jun. 2023) [[Paper]](https://arxiv.org/abs/2306.11644) [[Code]](https://github.com/kyegomez/phi-1)
  - The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only (NeurIPS 2023) [[Paper]](https://www.semanticscholar.org/paper/7a1e71cb1310c4a873e7a4e54d1a6dab0553adce) [[Dataset]](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
  - Textbooks Are All You Need II: phi-1.5 technical report (Arxiv, Sep. 2023) [[Paper]](https://arxiv.org/abs/2309.05463) [[Model]](https://huggingface.co/microsoft/phi-1_5)
  - When less is more: Investigating Data Pruning for Pretraining LLMs at Scale (Arxiv, Sep. 2023) [[Paper]](https://arxiv.org/abs/2309.04564)
  - Ziya2: Data-centric Learning is All LLMs Need (Arxiv, Nov. 2023) [[Paper]](https://arxiv.org/abs/2311.03301) [[Model]](https://huggingface.co/IDEA-CCNL/Ziya2-13B-Base)
  - Phi-2: The surprising power of small language models (Blog post, Dec. 2023) [[Post]](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
  - QuRating: Selecting High-Quality Data for Training Language Models (ICML 2024) [[Paper]](https://arxiv.org/abs/2402.09739) [[Code]](https://github.com/princeton-nlp/QuRating)

- #### Deduplication
  
  - Deduplicating training data makes language models better (ACL 2022) [[Paper]](https://arxiv.org/abs/2107.06499) [[Code]](https://github.com/google-research/deduplicate-text-datasets)
  - Deduplicating training data mitigates privacy risks in language models (ICML 2022) [[Paper]](https://proceedings.mlr.press/v162/kandpal22a.html) 
  - Noise-Robust De-Duplication at Scale (ICLR 2022) [[Paper]](https://scholar.harvard.edu/dell/publications/noise-robust-de-duplication-scale) 
  - SemDeDup: Data-efficient learning at web-scale through semantic deduplication (Arxiv, Mar. 2023) [[Paper]](https://arxiv.org/abs/2303.09540) [[Code]](https://github.com/facebookresearch/SemDeDup)

- #### Toxicity Filtering

  - Detoxifying language models risks marginalizing minority voices (NAACL-HLT, 2021) [[Paper]](https://arxiv.org/abs/2104.06390) [[Code]](https://github.com/albertkx/detoxifying-lms)
  - Challenges in detoxifying language models (EMNLP Findings, 2021) [[Paper]](https://arxiv.org/abs/2109.07445) 
  - What’s in the box? a preliminary analysis of undesirable content in the Common Crawl corpus (Arxiv, May 2021) [[Paper]](https://arxiv.org/abs/2105.02732) [[Code]](https://github.com/josephdviviano/whatsinthebox)
  - A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.13169)
 
- #### Diversity & Age

  - Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data (Arxiv, Jun. 2023) [[Paper]](https://arxiv.org/abs/2306.13840) 
  - D2 Pruning: Message Passing for Balancing Diversity and Difficulty in Data Pruning (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.07931) [[Code]](https://github.com/adymaharana/d2pruning)
  - A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.13169)

- #### *Social Biases

  - Documenting large webtext corpora: A case study on the Colossal Clean Crawled Corpus (EMNLP 2021) [[Paper]](https://arxiv.org/abs/2104.08758) 
  - An empirical survey of the effectiveness of debiasing techniques for pre-trained language models (ACL, 2022) [[Paper]](https://arxiv.org/abs/2110.08527) [[Code]](https://github.com/McGill-NLP/bias-bench)
  - Whose language counts as high quality? Measuring language ideologies in text data selection (EMNLP, 2022) [[Paper]](https://arxiv.org/abs/2201.10474) [[Code]](https://github.com/kernelmachine/quality-filter)
  - From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models (ACL 2023) [[Paper]](https://arxiv.org/abs/2305.08283) [[Code]](https://github.com/BunsenFeng/PoliLean)
 
- #### *Hallucinations
  
  - How Pre-trained Language Models Capture Factual Knowledge? A Causal-Inspired Analysis (ACL 2022) [[Paper]](https://arxiv.org/abs/2203.16747)
  - On the Origin of Hallucinations in Conversational Models: Is it the Datasets or the Models? (NAACL 2022) [[Paper]](https://arxiv.org/abs/2204.07931)
  - Sources of Hallucination by Large Language Models on Inference Tasks (EMNLP Findings, 2023) (https://arxiv.org/abs/2305.14552)

### Relations Among Different Aspects

- A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.13169)
- SlimPajama-DC: Understanding Data Combinations for LLM Training (Arxiv, Sep. 2023) [[Paper]](https://arxiv.org/abs/2309.10818) [[Model]](https://huggingface.co/MBZUAI-LLM) [[Dataset]](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- DeepSeek LLM: Scaling Open-Source Language Models with Longtermism (Arxiv, Jan. 2024) [[Paper]](https://arxiv.org/abs/2401.02954) [[Model]](https://github.com/deepseek-ai/DeepSeek-LLM)
- Scaling Laws for Data Filtering -- Data Curation cannot be Compute Agnostic (CVPR 2024) [[Paper]](https://arxiv.org/abs/2404.07177) [[Code]](https://github.com/locuslab/scaling_laws_data_filtering)
- Data Mixing Made Efficient: A Bivariate Scaling Law for Language Model Pretraining (Arxiv, May 2024) [[Paper]](https://arxiv.org/abs/2405.14908)

## Supervised Fine-Tuning

### Task composition

- Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ Tasks (EMNLP 2022) [[Paper]](https://arxiv.org/abs/2204.07705) [[Dataset]](https://github.com/allenai/natural-instructions)
- Finetuned Language Models Are Zero-Shot Learners (ICLR 2022) [[Paper]](https://arxiv.org/abs/2109.01652) [[Dataset]](https://github.com/google-research/FLAN)
- Multitask Prompted Training Enables Zero-Shot Task Generalization (ICLR 2022) [[Paper]](https://arxiv.org/abs/2110.08207) [[Code]](https://github.com/bigscience-workshop/t-zero)
- Scaling Instruction-Finetuned Language Models (Arxiv, Oct. 2022) [[Paper]](https://arxiv.org/abs/2210.11416) [[Dataset]](https://github.com/google-research/FLAN)
- OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization (Arxiv, Dec. 2022)  [[Paper]](https://arxiv.org/abs/2212.12017) [[Model]](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT-IML)
- The Flan Collection: Designing Data and Methods for Effective Instruction Tuning (ICML, 2023) [[Paper]](https://arxiv.org/abs/2301.13688) [[Dataset]](https://github.com/google-research/FLAN)
- Exploring the Benefits of Training Expert Language Models over Instruction Tuning (ICML, 2023) [[Paper]](https://arxiv.org/abs/2302.03202) [[Code]](https://github.com/joeljang/ELM)
- Data-Efficient Finetuning Using Cross-Task Nearest Neighbors (ACL Findings, 2023) [[Paper]](https://arxiv.org/abs/2212.00196) [[Code]](https://github.com/allenai/data-efficient-finetuning)
- Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.09246) 
- How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources (Arxiv, Jun. 2023) [[Paper]](https://arxiv.org/abs/2306.04751) [[Code]](https://github.com/allenai/open-instruct)
- How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.05492)
- LESS: Selecting Influential Data for Targeted Instruction Tuning (Arxiv, Feb. 2024)[[Paper]](https://arxiv.org/abs/2402.04333) [[Code]](https://github.com/princeton-nlp/LESS)
- Instruction Matters, a Simple yet Effective Task Selection Approach in Instruction Tuning for Specific Tasks (Arxiv, Apr. 2024) [[Paper]](https://arxiv.org/abs/2404.16418)

### Data Quality

- #### Instruction Quality

  - Self-refine: Iterative refinement with self-feedback (Arxiv, Mar. 2023) [[Paper]](https://arxiv.org/abs/2303.17651)[[Project]](https://selfrefine.info/)
  - Lima: Less is more for alignment (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.11206) [[Dataset]](https://huggingface.co/datasets/GAIR/lima)
  - Enhancing Chat Language Models by Scaling High-quality Instructional Conversations (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.14233) [[Code]](https://github.com/thunlp/UltraChat)
  - SelFee: Iterative Self-Revising LLM Empowered by Self-Feedback Generation (Blog post, May 2023) [[Project]](https://kaistai.github.io/SelFee/)
  - INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models (Arxiv, Jun. 2023) [[Paper]](https://arxiv.org/abs/2306.04757) [[Code]](https://github.com/declare-lab/instruct-eval)
  - Instruction mining: High-quality instruction data selection for large language models (Arxiv, Jul. 2023) [[Paper]](https://arxiv.org/abs/2307.06290) [[Code]](https://openreview.net/attachment?id=kce6LTZ5vY&name=supplementary_material)
  - AlpaGasus: Training A Better Alpaca with Fewer Data (Arxiv, Jul. 2023) [[Paper]](https://arxiv.org/abs/2307.08701) 
  - Harnessing the Power of David against Goliath: Exploring Instruction Data Generation without Using Closed-Source Models (Arxiv, Aug. 2023) [[Paper]](https://arxiv.org/abs/2308.12711) 
  - Self-Alignment with Instruction Backtranslation (Arxiv. Aug. 2023) [[Paper]](https://arxiv.org/abs/2308.06259) 
  - SELF: Language-Driven Self-Evolution for Large Language Models (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.00533)
  - LoBaSS: Gauging Learnability in Supervised Fine-tuning Data (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.13008) 
  - Tuna: Instruction Tuning using Feedback from Large Language Models (EMNLP 2023) [[Paper]](https://arxiv.org/abs/2310.13385) [[Code]](https://github.com/microsoft/LMOps)
  - Automatic Instruction Optimization for Open-source LLM Instruction Tuning (Arxiv, Nov. 2023) [[Paper]](https://arxiv.org/abs/2311.13246) [[Code]](https://github.com/lunyiliu/CoachLM)
  - MoDS: Model-oriented Data Selection for Instruction Tuning (Arxiv, Nov. 2023) [[Paper]](https://arxiv.org/abs/2311.15653) [[Code]](https://github.com/CASIA-LM/MoDS)
  - One Shot Learning as Instruction Data Prospector for Large Language Models (Arxiv, Dec. 2023) [[Paper]](https://arxiv.org/abs/2312.10302)
  - An Experimental Design Framework for Label-Efficient Supervised Finetuning of Large Language Models (Arxiv, Jan. 2024) [[Paper]](https://arxiv.org/abs/2401.06692)
  - Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning (Arxiv, Feb. 2024) [[Paper]](https://arxiv.org/abs/2402.00530) [[Code]](https://github.com/tianyi-lab/Superfiltering)
  - SelectIT: Selective Instruction Tuning for Large Language Models via Uncertainty-Aware Self-Reflection (Arxiv, Feb. 2024) [[Paper]](https://arxiv.org/abs/2402.16705) [[Code]](https://github.com/Blue-Raincoat/SelectIT)
  - From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning (NAACL 2024) [[Paper]](https://arxiv.org/abs/2308.12032) [[Code]](https://github.com/MingLiiii/Cherry_LLM)
  - Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning (ACL Findings 2024) [[Paper]](https://arxiv.org/abs/2402.10110) [[Code]](https://github.com/tianyi-lab/Reflection_Tuning)
  - Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models (Arxiv, Feb. 2024) [[Paper]](https://arxiv.org/abs/2402.10430)
  - SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models (Arxiv, Mar. 2024) [[Paper]](https://arxiv.org/abs/2403.07384)
  - Automated Data Curation for Robust Language Model Fine-Tuning (Arxiv, Mar. 2024) [[Paper]](https://arxiv.org/abs/2403.12776)
  - SHED: Shapley-Based Automated Dataset Refinement for Instruction Fine-Tuning (Arxiv, May 2024) [[Paper]](https://arxiv.org/abs/2405.00705)

- #### Instruction Diversity

  - Self-instruct: Aligning language models with self-generated instructions (ACL 2023) [[Paper]](https://arxiv.org/abs/2212.10560)[[Code]](https://github.com/yizhongw/self-instruct)
  - Stanford Alpaca (Mar. 2023) [[Code]](https://github.com/tatsu-lab/stanford_alpaca) 
  - Enhancing Chat Language Models by Scaling High-quality Instructional Conversation  (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.14233) [[Code]](https://github.com/thunlp/UltraChat)
  - Lima: Less is more for alignment (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.11206) [[Dataset]](https://huggingface.co/datasets/GAIR/lima)
  - #InsTag: Instruction Tagging for Analyzing Supervised Fine-Tuning of Large Language Models (Arxiv, Aug. 2023) [[Paper]](https://arxiv.org/abs/2308.07074) [[Code]](https://github.com/OFA-Sys/InsTag)
  - Explore-Instruct: Enhancing Domain-Specific Instruction Coverage through Active Exploration (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.09168) [[Code]](https://github.com/fanqiwan/Explore-Instruct)
  - DiffTune: A Diffusion-Based Approach to Diverse Instruction-Tuning Data Generation (NeurIPS 2023) [[Paper]](https://neurips2023-enlsp.github.io/papers/paper_82.pdf)
  - Self-Evolved Diverse Data Sampling for Efficient Instruction Tuning (Arxiv, Nov. 2023) [[Paper]](https://arxiv.org/abs/2311.08182) [[Code]](https://github.com/OFA-Sys/DiverseEvol)
  - Data Diversity Matters for Robust Instruction Tuning (Arxiv, Nov. 2023) [[Paper]](https://arxiv.org/abs/2311.14736)
  - Clustering and Ranking: Diversity-preserved Instruction Selection through Expert-aligned Quality Estimation (Arxiv, Feb. 2024) [[Paper]](https://arxiv.org/abs/2402.18191) [[Code]]([http://process.in/](https://github.com/IronBeliever/CaR))
  - Multi-view fusion for instruction mining of large language model (Information Fusion Oct. 2024) [[Paper]](https://www.sciencedirect.com/science/article/pii/S1566253524002586)

- #### Instruction Complexity

  - WizardLM: Empowering Large Language Models to Follow Complex Instructions (Arxiv, April 2023) [[Paper]](https://arxiv.org/abs/2304.12244) [[Code]](https://github.com/nlpxucan/WizardLM)
  - WizardCoder: Empowering Code Large Language Models with Evol-Instruct (Arxiv, Jun. 2023) [[Paper]](https://arxiv.org/abs/2306.08568) [[Code]](https://github.com/nlpxucan/WizardLM)
  - Orca: Progressive Learning from Complex Explanation Traces of GPT-4 (Arxiv, Jun. 2023) [[Paper]](https://arxiv.org/abs/2306.02707) [[Code]](https://github.com/Agora-X/Orca)
  - A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment (Arxiv, Aug. 2023) [[Paper]](https://arxiv.org/abs/2308.05696) 
  - #InsTag: Instruction Tagging for Analyzing Supervised Fine-Tuning of Large Language Models (Arxiv, Aug. 2023) [[Paper]](https://arxiv.org/abs/2308.07074) [[Code]](https://github.com/OFA-Sys/InsTag)
  - Can Large Language Models Understand Real-World Complex Instructions? (Arxiv, Sep. 2023) [[Paper]](https://arxiv.org/abs/2309.09150) [[Benchmark]](https://github.com/Abbey4799/CELLO)
  - Followbench: A multi-level fine-grained constraints following benchmark for large language models (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.20410) [[Code]](https://github.com/YJiangcm/FollowBench)
  - Conifer: Improving Complex Constrained Instruction-Following Ability of Large Language Models (Arxiv, Feb. 2024) [[Paper]](https://arxiv.org/abs/2404.02823) [[Code]](https://www.github.com/ConiferLM/Conifer)
  - From Complex to Simple: Enhancing Multi-Constraint Complex Instruction Following Ability of Large Language Models (Arxiv, Apr. 2024) [[Paper]](https://arxiv.org/abs/2404.15846) [[Code]](https://github.com/meowpass/FollowComplexInstruction)


- #### *Prompt Design

  - Reframing instructional prompts to gptk’s language (ACL Findings, 2022) [[Paper]](https://arxiv.org/abs/2109.07830v3) [[Code]](https://github.com/allenai/reframing)
  - Prompt Waywardness: The Curious Case of Discretized Interpretation of Continuous Prompts (NAACL, 2022) [[Paper]](https://arxiv.org/abs/2112.08348) [[Code]](https://github.com/alrope123/prompt-waywardness)
  - Demystifying Prompts in Language Models via Perplexity Estimation (Arxiv, Dec. 2022) [[Paper]](https://arxiv.org/abs/2212.04037) 
  - Did You Read the Instructions? Rethinking the Effectiveness of Task Definitions in Instruction Learning (ACL, 2023) [[Paper]](https://arxiv.org/abs/2306.01150) [[Code]](https://github.com/fanyin3639/rethinking-instruction-effectiveness)
  - Do Models Really Learn to Follow Instructions? An Empirical Study of Instruction Tuning (ACL, 2023) [[Paper]](https://arxiv.org/abs/2305.11383) 
  - The False Promise of Imitating Proprietary LLMs (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.15717) 
  - Exploring Format Consistency for Instruction Tuning (Arxiv, Jul. 2023) [[Paper]](https://arxiv.org/abs/2307.15504) 
  - Mind the instructions: a holistic evaluation of consistency and interactions in prompt-based learning (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.13486) 
  - Dynamics of Instruction Tuning: Each Ability of Large Language Models Has Its Own Growth Pace (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.19651)
 
- #### *Hallucinations
  - Lima: Less is more for alignment (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.11206) [[Dataset]](https://huggingface.co/datasets/GAIR/lima)
  - AlpaGasus: Training A Better Alpaca with Fewer Data (Arxiv, Jul. 2023) [[Paper]](https://arxiv.org/abs/2307.08701) 
  - Instruction mining: High-quality instruction data selection for large language models (Arxiv, Jul. 2023) [[Paper]](https://arxiv.org/abs/2307.06290) [[Code]](https://openreview.net/attachment?id=kce6LTZ5vY&name=supplementary_material)
  - Platypus: Quick, Cheap, and Powerful Refinement of LLMs (NeurIPS 2023 Workshop) [[Paper]](https://arxiv.org/abs/2308.07317) [[Code]](https://platypus-llm.github.io/)
 
### Data Quantity

- Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases (Arxiv, Mar. 2023) [[Paper]](https://arxiv.org/abs/2303.14742) 
- Lima: Less is more for alignment (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.11206) [[Dataset]](https://huggingface.co/datasets/GAIR/lima)
- Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.09246) 
- Scaling Relationship on Learning Mathematical Reasoning with Large Language Models (Arxiv, Aug. 2023) [[Paper]](https://arxiv.org/abs/2308.01825) [[Code]](https://github.com/OFA-Sys/gsm8k-ScRel)
- How Abilities In Large Language Models Are Affected By Supervised Fine-Tuning Data Composition (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.05492) 
- Dynamics of Instruction Tuning: Each Ability of Large Language Models Has Its Own Growth Pace (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.19651)
- When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method (ICLR 2024) [[Paper]](https://arxiv.org/abs/2402.17193)

### Dynamic Data-Efficient Learning

- #### Training Affects Data
  
  - NLU on Data Diets: Dynamic Data Subset Selection for NLP Classification Tasks (SustaiNLP, 2023) [[Paper]](https://arxiv.org/abs/2306.03208)
  - Becoming self-instruct: introducing early stopping criteria for minimal instruct tuning (Arxiv, Jul. 2023) [[Paper]](https://arxiv.org/abs/2307.03692)
  - Active Instruction Tuning: Improving Cross-Task Generalization by Training on Prompt Sensitive Tasks (EMNLP 2023) [[Paper]](https://arxiv.org/abs/2311.00288) [[Code]](https://github.com/PlusLabNLP/Active-IT)

- #### Data Affects Training
  
  - Dynosaur: A Dynamic Growth Paradigm for Instruction-Tuning Data Curation (Arxiv, May 2023) [[Paper]](https://arxiv.org/abs/2305.14327) [[Code]](https://github.com/WadeYin9712/Dynosaur)
  - OpenChat: Advancing Open-source Language Models with Mixed-Quality Data (Arxiv, Sep. 2023) [[Paper]](https://arxiv.org/abs/2309.11235) [[Code]](https://github.com/imoneoi/openchat)
  - How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition (Arxiv, Oct. 2023) [[Paper]](https://arxiv.org/abs/2310.05492)
  - Contrastive post-training large language models on data curriculum (Arxiv, Oct. 2023)[[Paper]](https://arxiv.org/abs/2310.02263)
  - InsCL: A Data-efficient Continual Learning Paradigm for Fine-tuning Large Language Models with Instructions (NAACL 2024) [[Paper]](https://arxiv.org/abs/2403.11435)
  - Conifer: Improving Complex Constrained Instruction-Following Ability of Large Language Models (Arxiv, Feb. 2024) [[Paper]](https://arxiv.org/abs/2404.02823) [[Code]](https://www.github.com/ConiferLM/Conifer)
  - Strategic Data Ordering: Enhancing Large Language Model Performance through Curriculum Learning (Arxiv, May 2024) [[Paper]](https://arxiv.org/abs/2405.07490)
 
### Relations Among Different Aspects

  - #InsTag: Instruction Tagging for Analyzing Supervised Fine-Tuning of Large Language Models (Arxiv, Aug. 2023) [[Paper]](https://arxiv.org/abs/2308.07074) [[Code]](https://github.com/OFA-Sys/InsTag)
  - Data Diversity Matters for Robust Instruction Tuning (Arxiv, Nov. 2023) [[Paper]](https://arxiv.org/abs/2311.14736)
  - Rethinking the Instruction Quality: LIFT is What You Need (Arxiv, Dec. 2023) [[Paper]](https://arxiv.org/abs/2312.11508)

## Useful Resources

- Practical guides for LLM [[Repo]](https://github.com/Mooler0410/LLMsPracticalGuide#practical-guide-for-data)
- Introduction to LLM [[Repo]](https://github.com/datainsightat/introduction_llm?search=1)
- Survey of LLM [[Repo]](https://github.com/RUCAIBox/LLMSurvey)
- Data-centric AI [[Repo]](https://github.com/daochenzha/data-centric-AI#prompt-engineering)
- Scaling laws for LLM [[Repo]](https://github.com/RZFan525/Awesome-ScalingLaws)
- Instruction datasets [[Repo]](https://github.com/yaodongC/awesome-instruction-dataset#the-multi-modal-instruction-datasets)
- Instruction tuning [[Repo1]](https://github.com/zhilizju/Awesome-instruction-tuning) [[Repo2]](https://github.com/SinclairCoder/Instruction-Tuning-Papers)
