# MELoRA Demo for CS221.Q21.KHTN

Day la ma nguon demo mon hoc **CS221.Q21.KHTN** cho bai toan fine-tuning hieu qua tham so bang **LoRA** va **MELoRA** tren cac tac vu NLP, dac biet la NER/token classification.

Repo duoc xay dung dua tren paper tham khao:

> **Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning**  
> Pengjie Ren, Chengshun Shi, Shiguang Wu, Mengqi Zhang, Zhaochun Ren, Maarten de Rijke, Zhumin Chen, Jiahuan Pei.  
> arXiv:2402.17263, 2024.

MELoRA la phuong phap mo rong LoRA bang cach dong bang trong so cua pretrained model va huan luyen mot nhom mini-LoRA. Thay vi chi dung mot adapter low-rank, MELoRA dung nhieu adapter nho de tao tinh da dang trong ensemble, qua do ky vong cai thien kha nang tong quat hoa trong khi van giu so tham so huan luyen o muc thap.

## Method Overview

<div align="center">
  <img src="./figs/method.png">
</div>

## Cau truc chinh

| File | Muc dich |
| --- | --- |
| `peft-0.5.0/` | Ban PEFT tuy chinh co ho tro MELoRA. Can cai dat editable truoc khi chay. |
| `run_*_lora.py`, `run_*_phobert.py` | Script Python huan luyen/evaluate/predict cho tung dataset. |
| `*_finetune.sh`, `*_phobert.sh` | Script bash cau hinh san thuc nghiem LoRA baseline va MELoRA. |
| `figs/method.png` | Hinh minh hoa phuong phap MELoRA. |
| `requirements.txt` | Thu vien Python can cai dat. |

## Cai dat moi truong

Khuyen nghi dung Python 3.10.

```bash
conda create -n MELoRA python=3.10
conda activate MELoRA

pip install torch==2.0.1
pip install -r requirements.txt

cd peft-0.5.0
pip install -e .
cd ..
```

Neu chay cac model/dataset tren Hugging Face lan dau, may can ket noi internet de tai model va dataset ve cache.

## Cach chay nhanh demo

Cac script bash hien dang dat:

- `WANDB_MODE=offline`: log W&B o che do offline.
- `seed=42`: co dinh seed de tai lap ket qua.
- `mode=base`: LoRA baseline.
- `mode=me` hoac `mode=melora`: MELoRA.
- `rank=8, l_num=1`: cau hinh LoRA baseline trong da so script encoder.
- `rank=8, l_num=2`: cau hinh MELoRA trong da so script encoder.
- `target_modules="query value"`: gan adapter vao cac module attention query/value cua RoBERTa/PhoBERT.

Chay tu thu muc goc cua repo:

```bash
bash vietmed_ner_phobert.sh
```

Script tren se chay lan luot LoRA baseline va MELoRA cho VietMed-NER. Ket qua duoc ghi vao thu muc output tuong ung, vi du:

```text
./phobert-large-vietmed-ner/<run_name>/model
./phobert-large-vietmed-ner/<run_name>/model/test_predictions_full.jsonl
```

## Danh sach script thuc nghiem

| Script | Tac vu | Model mac dinh | Dataset | Cau hinh chinh |
| --- | --- | --- | --- | --- |
| `vietmed_ner_phobert.sh` | Vietnamese medical NER | `vinai/phobert-large` | `leduckhai/VietMed-NER` | 40 epochs, batch 64, max train 200000, F1 |
| `udd_phobert.sh` | Vietnamese POS/token classification | `vinai/phobert-base` | `undertheseanlp/UDD-v0.1` | label `upos`, 40 epochs, batch 64, max train 5000, F1 |
| `students_feedback_phobert.sh` | Vietnamese student feedback classification | `vinai/phobert-base` | `uitnlp/vietnamese_students_feedback` | label `sentiment`, 40 epochs, batch 64, max train 1500, F1 |
| `bc5cdr_ner_finetune.sh` | Biomedical NER | `FacebookAI/roberta-base` | `tner/bc5cdr` | 10 epochs, batch 64, max train 5000, F1 |
| `coco_finetune.sh` | COVID conspiracy classification | `FacebookAI/roberta-base` | `Jlangguth/COCO` | 10 epochs, batch 64, max train 5000, F1 |
| `sarcasm_detection_finetune.sh` | Sarcasm detection | `FacebookAI/roberta-base` | `Heschmat/news-headlines-dataset-sarcasm-detection` | 10 epochs, batch 64, max train 1000, F1 |
| `glue_finetune.sh` | GLUE NLU benchmark | `FacebookAI/roberta-base` | GLUE | mac dinh dang chay `stsb`, `cola` LoRA baseline |
| `llama_finetune.sh` | Instruction tuning | `meta-llama/Llama-2-7b-hf` | template/data trong script Python | LoRA rank 64 va MELoRA rank 1, `lora_n=8` |

## Lenh chay tung thuc nghiem

```bash
# Vietnamese medical NER
bash vietmed_ner_phobert.sh

# Vietnamese POS/token classification
bash udd_phobert.sh

# Vietnamese students feedback sentiment classification
bash students_feedback_phobert.sh

# Biomedical NER tren BC5CDR
bash bc5cdr_ner_finetune.sh

# COVID conspiracy classification
bash coco_finetune.sh

# Sarcasm detection
bash sarcasm_detection_finetune.sh

# GLUE benchmark
bash glue_finetune.sh

# Instruction tuning voi LLaMA
bash llama_finetune.sh
```

## Tuy chinh thuc nghiem

Truoc khi chay, co the sua truc tiep trong file `.sh`:

- `model_name_or_path`: doi model Hugging Face hoac duong dan model local.
- `dataset_name`: doi dataset Hugging Face.
- `max_train_samples`: gioi han so mau train de chay demo nhanh hon.
- `num_train_epochs`, `learning_rate`, `batch_size`: cau hinh huan luyen.
- `rank`: rank cua LoRA hoac rank cua tung mini-LoRA.
- `l_num`: so luong mini-LoRA khi dung MELoRA.
- `wandb_project`: ten project W&B. Mac dinh trong script la `project_name`.
- `CUDA_VISIBLE_DEVICES`: GPU dung de chay, trong cac script RoBERTa dang dat la `0`.

Voi GLUE, file `glue_finetune.sh` hien chi chay LoRA baseline cho:

```bash
task_base=('stsb' 'cola')
```

Neu muon chay MELoRA, bo comment dong:

```bash
# run $task "8" "2" "me"
```

Voi `llama_finetune.sh`, can dam bao co quyen truy cap model `meta-llama/Llama-2-7b-hf` hoac thay `--base_model` bang model local/phu hop voi may demo.

## Output

Moi lan chay tao mot thu muc rieng theo `run_name`. Thu muc nay thuong gom:

- `model/`: checkpoint/model tot nhat duoc luu boi Hugging Face Trainer.
- `log/`: log training, neu script co truyen `--logging_dir`.
- `test_predictions_full.jsonl`: chi tiet du doan tren test set voi cac script co `--do_predict`.
- log W&B offline, neu `--report_to wandb` duoc bat.

Metric chinh cua cac demo NER/classification la **F1**. GLUE dung metric rieng theo task, vi du `pearson` cho STS-B va `matthews_correlation` cho CoLA.

## Ghi chu cho demo CS221

Kich ban demo nen tap trung vao:

1. Gioi thieu bai toan parameter-efficient fine-tuning: pretrained model duoc dong bang, chi adapter duoc train.
2. So sanh LoRA baseline voi MELoRA qua hai cau hinh trong script: `base` va `me`.
3. Chay mot script NER tieu bieu, uu tien `vietmed_ner_phobert.sh` hoac `bc5cdr_ner_finetune.sh`.
4. Trinh bay output: F1, checkpoint, file du doan chi tiet `test_predictions_full.jsonl`.
5. Lien he voi paper MELoRA: mini-ensemble giup tang da dang adapter trong khi van tiet kiem tham so.

## Thanks

Code duoc phat trien dua tren:

- [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)
- [huggingface/peft](https://github.com/huggingface/peft)
- [huggingface/transformers](https://github.com/huggingface/transformers)

## Cite

Neu su dung method/code nay, vui long cite:

```bibtex
@article{melora,
  title={Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning},
  author={Ren, Pengjie and Shi, Chengshun and Wu, Shiguang and Zhang, Mengqi and Ren, Zhaochun and de Rijke, Maarten and Chen, Zhumin and Pei, Jiahuan},
  journal={arXiv preprint arXiv:2402.17263},
  year={2024}
}
```
