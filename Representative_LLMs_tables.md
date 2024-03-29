# Representative LLMs and Their Data Management Strategies


Table 1. The data management strategies used by representative pretrained models. The blank units mean no specific design of corresponding strategies according to the original papers. The '-' means that the data management process is not released. Part of the data is adopted from [work of Longpre et al.](https://arxiv.org/abs/2305.13169).
| pretrained LLMs | Open-souced | Quantity    | Deduplication                 | Quality Filters| Toxicity Filters | Domian Composition|
| --------------- | ----------- | ----------- | ----------------------------- | --------------------- | ---------------- | ------------------------------------------------------------ |
| T5| $\surd$| ~750GB| N-gram| Heuristic| Heuristic| 99% Web, < 1% Wiki|
| GPT-3|| 499B tokens | MinHashLSH| Classifier|| 82% Web, 16% Books, 3% Wiki|
| GLaM|| 1.6T tokens || Classifier || 46% Web, 28% Dialog, 20% Books, 6% Wiki|
| LaMDA|| 1.56T words ||| | 50% Dialog, 25% Web, 12.5% Wiki, 12.5% Code|
| Chinchilla || 1.4T tokens | N-gram, Document-level| Heuristic| Heuristic| 65% Web, 30% Books, 4% Code, 1% Wiki|
| AlphaCode|| 715.1GB| Document-level| Heuristic|| 100% Code|
| GLM| $\surd$| 400B tokens |||| 50% Pile, 50% Chinese Web data|
| BLOOM| $\surd$| 1.61TB text | SimHash, Substring clustering | Heuristic| Heuristic| 60% Web, 10% Books, 10% Code, 10% Academic, 5% Dialog, 5% Wiki |
| PaLM|| 780B tokens | Levenshtein distance| Heuristic, Classifier | Classifier| 50% Dialog, 28% Web, 13% Books, 5% Code, 4% Wiki|
| LLaMA| $\surd$| 1.4T tokens | Line-level, book-level| Heuristic, Classifier | Classifier| 82% Web, 4.5% Code, 4.5% Wiki, 4.5% Books, 2.5% Academic, 2% Dialog |
| Mistral| $\surd$| - | - | - | - | - |
| phi-1/1.5 | $\surd$| ~7B tokens  || Classifier|| 99% Academic, <1% Code|
| phi-2 | $\surd$ | 1.4B tokens || Classifier|||
| GPT-4 || - | - | - | - | - |
| LLaMA 2| $\surd$ | 2.0T tokens ||| Heuristic ||
| QWen | $\surd$| 3T tokens | Exact Match, MinHash, LHS | Heuristic, Classifier | Classifier | Web, Books, Codes, Academic |
| Deepseek LLM | $\surd$ | - | - | - | - | - |


Table 2. The data management strategies used by representative supervised finetuned models. The blank units mean no specific design of corresponding strategies according to the original papers.

 | Supervised Finetuned LLMs | Base Model | Dataset Proposed/Used| Quantity | Quality Control | Diversity Control  | Complexity Enhancing | # of Tasks | Task Balancing |
| --------------------------------- | ---------- | -------------------------------------------------- | -------- | ------------------------ | ------------------ | ---------------------------- | ---------- | -------------------------- |
| Tk-Instruct | T5-LM | Super-NaturalInstructions | 5M | Heuristic, Human||| 1616 | Limited instances per task |
| Flan-T5 | T5-LM | Flan 2022 | 15M || Input Inversion || 1836 | Experiments, intuitions |
| OPT-IML| OPT | OPT-IML Bench | 18M |||| 2000| Experiments |
| Alpaca| LLaMA | Alpaca | 52K | Heuristic | ROUGE-L similarity || 80||
| Vicuna | LLaMA | ShareGPT | 70K | Heuristic ||||
| LIMA | LLaMA | LIMA | 1K| Heuristic, Human| Heuristic, Human||||
| Dolly | pythia | databricks-dolly-15k | 15K | Human |||||
| Orca | LLaMA | sampled Flan 2022 with Chat-GPT/GPT-4 augmentation | 5M ||| Chat-GPT/ GPT-4 augmentation ||                            |
| WizardLM/ WizardCoder/ WizardMath | LLaMA | WizardLM/ WizardCoder/ WizardMath | 250K || Evol-Instruct | Evol-Instruct |||
| AlpaGasus | LLaMA-1/2  | AlpaGasus | 9K | Chat-GPT grading |||||
| Platypus| LLaMA-2| Open-Platypus| 25K | Deduplication, Heuristic |||||
| OpenChat | LLaMA 2 | ShareGPT| 6K | C-RLFT |||||
| MAmmoTH  | LLaMA-2 | MathInstruct | 260K |||| 7 math fields | Combining CoT and PoT |
