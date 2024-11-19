# mt_luxembourgish

## Problematics

- The scarcity of Luxembourgish corpus in low-resource languages leads to insufficient training data. Translating Luxembourgish to English is relatively accurate, but the reverse is more challenging.  
- Pseudo-translated data sources do not yield high-quality translations but can serve as training data.  
- How to achieve better bidirectional translation performance with monolingual corpus and only a dictionary.  

## Methods  

1. Pseudo-translation augmentation.  
2. Utilizing related corpora, such as German or French, for enhancement.  
3. Knowledge distillation from GPT-4 to extract additional corpus data (from diverse sources: Meta, OpenAI, others).  
4. Exploiting ChatGPT's generative tendencies for data augmentation, combined with static methods for cross-check validation.  
5. Fine-tuning (FT) model selection, training methodology optimization, and discovering reverse translation techniques.  
6. Validating our data methodology on similar languages, e.g., Norwegian and Icelandic, to achieve comparable results.  

## Steps  

1. Data cleaning.  
2. Testing the performance of LLaMA 3.2 on French.  

## Code Explanations  

1. *data:* Contains all example datasets used in the experiments.
2. *models*: Stores pre-trained model cards , fine-tuned versions, and checkpoints.
3. *scripts*: Contains reusable Python scripts for preprocessing, training, evaluation, and translation tasks.
    * `bLocal_LLM_running_inference.py` is used for running huggingface local pipeline and do the inference.

    * `nllb_running_inference.py` is used for running nllb local pipeline and do the infernece of translation.

    * `*.sh` files are used for submitting jobs on HPC.
4. *notebook*: 
    * `data_chekcing.ipynb` is used for data checking the change it to csv
    * `data_procesing.ipynb` is used for processing the data: split, filter and language checking.
    * `model_inference.ipynb` is used for tesing the model itself.

5. *utils*: Utility functions and modules for common tasks



