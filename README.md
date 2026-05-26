# MELoRA Demo for CS221.Q21.KHTN

Đây là mã nguồn demo môn học **CS221.Q21.KHTN** cho bài toán fine-tuning hiệu quả tham số bằng **LoRA** và **MELoRA** trên các tác vụ NLP, đặc biệt là NER/token classification.


MELoRA là phương pháp mở rộng LoRA bằng cách đóng băng trọng số của pretrained model và huấn luyện một nhóm mini-LoRA. Thay vì chỉ dùng một adapter low-rank, MELoRA dùng nhiều adapter nhỏ để tạo tính đa dạng trong ensemble, qua đó kỳ vọng cải thiện khả năng tổng quát hóa trong khi vẫn giữ số tham số huấn luyện ở mức thấp.

## Method Overview

<div align="center">
  <img src="./figs/method.png">
</div>

## Cấu trúc chính

| File | Mục đích |
| --- | --- |
| `peft-0.5.0/` | Bản PEFT tùy chỉnh có hỗ trợ MELoRA. Cần cài đặt editable trước khi chạy. |
| `run_*_lora.py`, `run_*_phobert.py` | Script Python huấn luyện/evaluate/predict cho từng dataset. |
| `*_finetune.sh`, `*_phobert.sh` | Script bash cấu hình sẵn thí nghiệm LoRA baseline và MELoRA. |
| `figs/method.png` | Hình minh họa phương pháp MELoRA. |
| `requirements.txt` | Thư viện Python cần cài đặt. |

## Cài đặt môi trường

Khuyến nghị dùng Python 3.10.

```bash
conda create -n MELoRA python=3.10
conda activate MELoRA

pip install torch==2.0.1
pip install -r requirements.txt

cd peft-0.5.0
pip install -e .
cd ..
```

Nếu chạy các model/dataset trên Hugging Face lần đầu, máy cần kết nối internet để tải model và dataset về cache.

## Cách chạy nhanh demo

Các script bash hiện đang đặt:

- `WANDB_MODE=offline`: log W&B ở chế độ offline.
- `seed=42`: cố định seed để tái lập kết quả.
- `mode=base`: LoRA baseline.
- `mode=me` hoặc `mode=melora`: MELoRA.
- `rank=8, l_num=1`: cấu hình LoRA baseline trong đa số script encoder.
- `rank=8, l_num=2`: cấu hình MELoRA trong đa số script encoder.
- `target_modules="query value"`: gắn adapter vào các module attention query/value của RoBERTa/PhoBERT.

Chạy từ thư mục gốc của repo:

```bash
bash vietmed_ner_phobert.sh
```

Script trên sẽ chạy lần lượt LoRA baseline và MELoRA cho VietMed-NER. Kết quả được ghi vào thư mục output tương ứng, ví dụ:

```text
./phobert-large-vietmed-ner/<run_name>/model
./phobert-large-vietmed-ner/<run_name>/model/test_predictions_full.jsonl
```

## Danh sách script thí nghiệm

| Script | Tác vụ | Model mặc định | Dataset | Cấu hình chính |
| --- | --- | --- | --- | --- |
| `vietmed_ner_phobert.sh` | Vietnamese medical NER | `vinai/phobert-large` | `leduckhai/VietMed-NER` | 40 epochs, batch 64, max train 200000, F1 |
| `udd_phobert.sh` | Vietnamese POS/token classification | `vinai/phobert-base` | `undertheseanlp/UDD-v0.1` | label `upos`, 40 epochs, batch 64, max train 5000, F1 |
| `students_feedback_phobert.sh` | Vietnamese student feedback classification | `vinai/phobert-base` | `uitnlp/vietnamese_students_feedback` | label `sentiment`, 40 epochs, batch 64, max train 1500, F1 |
| `bc5cdr_ner_finetune.sh` | Biomedical NER | `FacebookAI/roberta-base` | `tner/bc5cdr` | 10 epochs, batch 64, max train 5000, F1 |
| `coco_finetune.sh` | COVID conspiracy classification | `FacebookAI/roberta-base` | `Jlangguth/COCO` | 10 epochs, batch 64, max train 5000, F1 |
| `sarcasm_detection_finetune.sh` | Sarcasm detection | `FacebookAI/roberta-base` | `Heschmat/news-headlines-dataset-sarcasm-detection` | 10 epochs, batch 64, max train 1000, F1 |
| `glue_finetune.sh` | GLUE NLU benchmark | `FacebookAI/roberta-base` | GLUE | mặc định đang chạy `stsb`, `cola` LoRA baseline |
| `llama_finetune.sh` | Instruction tuning | `meta-llama/Llama-2-7b-hf` | template/data trong script Python | LoRA rank 64 và MELoRA rank 1, `lora_n=8` |

## Lệnh chạy từng thí nghiệm

```bash
# Vietnamese medical NER
bash vietmed_ner_phobert.sh

# Vietnamese POS/token classification
bash udd_phobert.sh

# Vietnamese students feedback sentiment classification
bash students_feedback_phobert.sh

# Biomedical NER trên BC5CDR
bash bc5cdr_ner_finetune.sh

# COVID conspiracy classification
bash coco_finetune.sh

# Sarcasm detection
bash sarcasm_detection_finetune.sh

# GLUE benchmark
bash glue_finetune.sh

# Instruction tuning với LLaMA
bash llama_finetune.sh
```

## Tùy chỉnh thí nghiệm

Trước khi chạy, có thể sửa trực tiếp trong file `.sh`:

- `model_name_or_path`: đổi model Hugging Face hoặc đường dẫn model local.
- `dataset_name`: đổi dataset Hugging Face.
- `max_train_samples`: giới hạn số mẫu train để chạy demo nhanh hơn.
- `num_train_epochs`, `learning_rate`, `batch_size`: cấu hình huấn luyện.
- `rank`: rank của LoRA hoặc rank của từng mini-LoRA.
- `l_num`: số lượng mini-LoRA khi dùng MELoRA.
- `wandb_project`: tên project W&B. Mặc định trong script là `project_name`.
- `CUDA_VISIBLE_DEVICES`: GPU dùng để chạy, trong các script RoBERTa đang đặt là `0`.

Với GLUE, file `glue_finetune.sh` hiện chỉ chạy LoRA baseline cho:

```bash
task_base=('stsb' 'cola')
```

Nếu muốn chạy MELoRA, bỏ comment dòng:

```bash
# run $task "8" "2" "me"
```

Với `llama_finetune.sh`, cần đảm bảo có quyền truy cập model `meta-llama/Llama-2-7b-hf` hoặc thay `--base_model` bằng model local/phù hợp với máy demo.

## Output

Mỗi lần chạy tạo một thư mục riêng theo `run_name`. Thư mục này thường gồm:

- `model/`: checkpoint/model tốt nhất được lưu bởi Hugging Face Trainer.
- `log/`: log training, nếu script có truyền `--logging_dir`.
- `test_predictions_full.jsonl`: chi tiết dự đoán trên test set với các script có `--do_predict`.
- log W&B offline, nếu `--report_to wandb` được bật.

Metric chính của các demo NER/classification là **F1**. GLUE dùng metric riêng theo task, ví dụ `pearson` cho STS-B và `matthews_correlation` cho CoLA.


## Thanks

Code được phát triển dựa trên:

- [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)
- [huggingface/peft](https://github.com/huggingface/peft)
- [huggingface/transformers](https://github.com/huggingface/transformers)

## Cite

Nếu sử dụng method/code này, vui lòng cite:

```bibtex
@inproceedings{ren-etal-2024-melora,
    title = "{MEL}o{RA}: Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning",
    author = "Ren, Pengjie  and
      Shi, Chengshun  and
      Wu, Shiguang  and
      Zhang, Mengqi  and
      Ren, Zhaochun  and
      de Rijke, Maarten  and
      Chen, Zhumin  and
      Pei, Jiahuan",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.168/",
    doi = "10.18653/v1/2024.acl-long.168",
    pages = "3052--3064",
    abstract = "Parameter-efficient fine-tuning (PEFT) is a popular method for tailoring pre-trained large language models (LLMs), especially as the models' scale and the diversity of tasks increase. Low-rank adaptation (LoRA) is based on the idea that the adaptation process is intrinsically low-dimensional, i.e., significant model changes can be represented with relatively few parameters. However, decreasing the rank encounters challenges with generalization errors for specific tasks when compared to full-parameter fine-tuning. We present MELoRA, a mini-ensemble low-rank adapters that uses fewer trainable parameters while maintaining a higher rank, thereby offering improved performance potential.The core idea is to freeze original pretrained weights and train a group of mini LoRAs with only a small number of parameters. This can capture a significant degree of diversity among mini LoRAs, thus promoting better generalization ability. We conduct a theoretical analysis and empirical studies on various NLP tasks. Our experimental results show that, compared to LoRA, MELoRA achieves better performance with 8 times fewer trainable parameters on natural language understanding tasks and 36 times fewer trainable parameters on instruction following tasks, which demonstrates the effectiveness of MELoRA."
}
```
