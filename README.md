# MEDIQA 2021 – Consumer Health Question & Multi-Answer Summarization

**Team Project:** * Group 11: Advanced Neural Approaches for Extractive and Abstractive Text Summarization in Health Information Retrieval*

**Team Members:** 
- Rahul Kumar - 11859007 [@rk2903](https://huggingface.co/rk2903)
- Veera Venkata Megha Shyam Ankem - 11807512	
- Suguna Sai Navaneeth Rentala - 11800972
- Aditya Narayana Reddy Nallimilli - 11805743
  
---

## Project Resources

| Resource                                      | Link                                                                                                 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Google Drive (Models + Logs + Outputs)** | [Drive Folder](https://drive.google.com/drive/folders/1PGSBgiuRB8Biqq4AbCPvOmE2l1tLLdnU?usp=sharing) |
| **Hugging Face Profile**                   | [https://huggingface.co/rk2903](https://huggingface.co/rk2903)                                       |
| **T5 Model Checkpoint (Task 1)**           | [rk2903/t5_meqsum_summarizer](https://huggingface.co/rk2903/t5_meqsum_summarizer)                    |
| **Datasets (MeQSum & MEDIQA-AnS)**         | [rk2903/datasets](https://huggingface.co/rk2903/datasets)                                            |

---

## Tasks Overview

### **Task 1 – Question Summarization**

**Goal:** Generate a concise, expert-style summary of a long or multi-part consumer health question (CHQ).
**Dataset:** [MeQSum](https://github.com/abachaa/MeQSum) (+ NLM 2020 validation/test sets).

| Role             | Description                                |
| ---------------- | ------------------------------------------ |
| **Input**        | Full consumer health question              |
| **Output**       | Expert-written short summary (abstractive) |
| **Target Field** | `multi_abs_summ`                           |
| **Prefix**       | `"summarize: "`                            |

**Example**

```
Input:  summarize: My child has a fever and sore throat for 3 days. Could this be strep or a virus?
Output: Causes and treatments for fever and sore throat in children.
```

---

### **Task 2 – Multi-Answer Summarization**

**Goal:** Combine multiple retrieved medical answers for one CHQ into a single, coherent, expert summary.
**Dataset:** [MEDIQA-AnS](https://github.com/abachaa/MEDIQA2021-Tasks/tree/main/MEDIQA-ANS) or reconstructed CHiQA-based files.

| Role             | Description                                 |
| ---------------- | ------------------------------------------- |
| **Input**        | Question + multiple long answer passages    |
| **Output**       | Unified expert summary                      |
| **Target Field** | `multi_ans_summ` or `final_summary`         |
| **Prefix**       | `"multianswer: question: ... context: ..."` |

**Example**

```
Input:  multianswer: question: What are the symptoms and treatments for bronchitis?
        context: Acute bronchitis causes cough, mucus, and fatigue. [SEP]
                 Chronic bronchitis is long-term airway inflammation. [SEP]
                 Treatment includes rest, fluids, and bronchodilators.
Output: Bronchitis causes cough and mucus due to airway inflammation.
        Acute cases resolve with rest; chronic cases may need bronchodilators.
```

---

## Model Architecture

* **Base Model:** [`t5-base`](https://huggingface.co/t5-base)
* **Fine-Tuning Framework:** Hugging Face Transformers + Trainer API
* **Objective:** Minimize validation loss (cross-entropy)
* **Saved Checkpoint:** Best model based on lowest validation loss

**Evaluation Metrics:**
ROUGE-1 / ROUGE-2 / ROUGE-L • BLEU • BERTScore • Exact Match

---

## Data Preparation Summary

### Task 1

| Field                       | Use        |
| --------------------------- | ---------- |
| `question`                  | **Input**  |
| `multi_abs_summ`            | **Output** |
| `multi_ext_summ`, `answers` | Optional   |

### Task 2

| Field                              | Use                                 |
| ---------------------------------- | ----------------------------------- |
| `question`                         | **Input**                           |
| `answers` (list of texts)          | Concatenated with `[SEP]` separator |
| `multi_ans_summ` / `final_summary` | **Output**                          |

---

## Training & Evaluation

```bash
# install dependencies
pip install transformers datasets evaluate accelerate sentencepiece rouge_score sacrebleu bert-score

# run training (Task 1 or Task 2)
python train_t5_meqsum.py    # or train_t5_task2.py

# evaluate on test set
python evaluate_t5.py
```

**Training Parameters**

```
learning_rate = 5e-5
batch_size    = 8
num_epochs    = 8
max_source_len = 512 (Task 1) / 1024 (Task 2)
max_target_len = 64 (Task 1) / 128 (Task 2)
```

---

## Example Inference

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "rk2903/t5_meqsum_summarizer"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = "summarize: What are the causes and treatments of high cholesterol?"
input_ids = tokenizer(text, return_tensors="pt").input_ids
output_ids = model.generate(input_ids, max_length=64, num_beams=4)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## Results (Illustrative)

| Metric         | Task 1 (Question Summarization) | Task 2 (Multi-Answer Summarization) |
| -------------- | ------------------------------- | ----------------------------------- |
| ROUGE-1        | 0.45                            | 0.49                                |
| ROUGE-2        | 0.26                            | 0.31                                |
| ROUGE-L        | 0.44                            | 0.46                                |
| BLEU           | 0.22                            | 0.24                                |
| BERTScore (F1) | 0.86                            | 0.88                                |

*(Values shown for demonstration; actual results depend on fine-tuning run.)*

---

## Citation

1. Abacha, A. B., et al. (2021). 'Overview of the MEDIQA 2021 Shared Task on Summarization in the Medical Domain.' Proceedings of the 20th Workshop on Biomedical Language Processing. ACL 2021.
2. Google Research. 'Text-To-Text Transfer Transformer (T5): Exploring the Limits of Transfer Learning with a Unified Text-to-Text Framework.' JMLR, 2020.
3. MEDIQA GitHub Project. 'MEDIQA 2021 Shared Task Data.' National Library of Medicine (NLM), 2021.
4. Zhang et al., “PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization,” ICML 2020.
5. Mihalcea, R., & Tarau, P. (2004). “TextRank: Bringing Order into Texts.” EMNLP 2004.
---

## Contact

* Hugging Face: [@rk2903](https://huggingface.co/rk2903)
* Google Drive: [Project Files & Checkpoints](https://drive.google.com/drive/folders/1PGSBgiuRB8Biqq4AbCPvOmE2l1tLLdnU?usp=sharing)
