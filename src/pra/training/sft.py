from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from pra.utils.prompts import MU_PHI_SYSTEM as SYSTEM_PROMPT


@dataclass
class SFTConfig:
    reward_model_backbone_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset: str = "process-reward-agents/medpra-training-dataset-v2"
    output_root: str = "./outputs/sft"
    run_name: Optional[str] = None  # auto-generated if None
    timestamp: Optional[str] = None  # override via env TRAIN_TIMESTAMP for DDP

    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    max_len: int = 16384
    learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Data options
    train_subset_size: Optional[int] = None
    train_sample_filter: Optional[List[int]] = None
    use_is_correct_as_reasoning: bool = False
    docs_mode: str = "all_docs"
    no_docs_ratio: Optional[float] = None

    # Evaluation options
    eval_steps: int = 400
    eval_dev_size: Optional[int] = 1000
    eval_search_labels: bool = False

    # Checkpointing
    save_total_limit: int = 5
    save_only_model: bool = True
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_reasoning_auc_roc"
    greater_is_better: bool = True

    # W&B
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_enabled: bool = False

    def derive_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        docs_suffix = self.docs_mode + (
            str(self.no_docs_ratio) if self.docs_mode == "ratio" and self.no_docs_ratio is not None else ""
        )
        is_correct = "_iscorr" if self.use_is_correct_as_reasoning else ""
        ts = self.timestamp or os.environ.get("TRAIN_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = ts
        name = (
            f"{self.reward_model_backbone_path.split('/')[-1]}_SFT_{docs_suffix}{is_correct}"
            f"_bs{self.batch_size}x{self.gradient_accumulation_steps}"
            f"_lr{self.learning_rate}_{ts}"
        )
        self.run_name = name
        return name


def load_config(path: str, overrides: Optional[dict] = None) -> SFTConfig:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in raw.items()}
    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})
    valid = {f.name for f in SFTConfig.__dataclass_fields__.values()}
    unknown = sorted(set(raw) - valid)
    if unknown:
        print(f"[sft] Ignoring unknown config keys: {unknown}", file=sys.stderr)
    return SFTConfig(**{k: v for k, v in raw.items() if k in valid})


def _build_user_prompt(item: dict, include_docs: bool = True) -> str:
    options_text = (
        "\n".join(f"({k}): {v}" for k, v in item["options"].items()) if item.get("options") else ""
    )
    parts: List[str] = []
    if include_docs and item.get("related_docs"):
        parts.append(
            "=== DOCUMENTS ===\n"
            + "\n".join(f"Doc {i + 1}: {doc}" for i, doc in enumerate(item["related_docs"]))
        )
    parts.append(f"=== QUESTION ===\n{item['question']}\n{options_text}")
    parts.append(f"=== REASONING SOLUTION ===\n{item['reasoning_solution']}")
    return "\n\n".join(parts)


class MedPraDataset(Dataset):
    def __init__(
        self,
        hf_split,
        tokenizer,
        max_length: int,
        docs_mode: str = "all_docs",
        no_docs_ratio: Optional[float] = None,
        use_is_correct_as_reasoning: bool = False,
    ):
        if docs_mode not in ("all_docs", "ratio"):
            raise ValueError(f"Unknown docs_mode {docs_mode!r}; expected 'all_docs' or 'ratio'")
        if docs_mode == "ratio" and no_docs_ratio is None:
            raise ValueError("docs_mode='ratio' requires no_docs_ratio")

        self.dataset = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.docs_mode = docs_mode
        self.use_is_correct_as_reasoning = use_is_correct_as_reasoning

        if docs_mode == "ratio":
            rng = np.random.default_rng(42)
            self._include_docs_mask = rng.random(len(self.dataset)) >= no_docs_ratio
        else:
            self._include_docs_mask = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        include_docs = (
            bool(self._include_docs_mask[idx])
            if self._include_docs_mask is not None
            else True
        )

        reasoning_label = (
            1 if self.use_is_correct_as_reasoning and item["is_correct"]
            else item["reasoning_label"]
        )
        search_label = item["search_label"]

        user_prompt = _build_user_prompt(item, include_docs=include_docs)
        chat_input = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = f"{reasoning_label},{search_label}"
        full_text = chat_input + response

        enc = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.max_length)
        prompt_enc = self.tokenizer(chat_input, return_tensors="pt", truncation=True, max_length=self.max_length)
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = enc["input_ids"].clone()
        labels[0, :prompt_len] = -100  # mask prompt tokens

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "reasoning_label": torch.tensor(reasoning_label, dtype=torch.long),
            "search_label": torch.tensor(search_label, dtype=torch.long),
        }


class _PaddedCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {}
        for key in ("input_ids", "attention_mask", "labels"):
            seqs = [f[key] for f in features]
            pad = 0 if "mask" in key else (-100 if "labels" in key else self.tokenizer.pad_token_id)
            batch[key] = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad)
        for key in ("reasoning_label", "search_label"):
            batch[key] = torch.stack([f[key] for f in features])
        return batch


def _classification_metrics(y_true, y_pred, prefix: str = "") -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    out = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tn = fp = fn = tp = 0
    out[f"{prefix}npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    out[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return out


def _build_compute_metrics(eval_search: bool):
    from sklearn.metrics import roc_auc_score

    def compute_metrics(eval_pred) -> dict:
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        pred_reasoning = np.clip(preds[:, 0], 0, 1).round().astype(int)
        pred_search = np.clip(preds[:, 1], 0, 1).round().astype(int)
        prob_reasoning = np.clip(preds[:, 2], 0, 1)
        prob_search = np.clip(preds[:, 3], 0, 1)
        true_reasoning = np.clip(labels[:, 0], 0, 1).round().astype(int)
        true_search = np.clip(labels[:, 1], 0, 1).round().astype(int)

        metrics = _classification_metrics(true_reasoning, pred_reasoning, prefix="reasoning_")
        metrics["reasoning_auc_roc"] = (
            roc_auc_score(true_reasoning, prob_reasoning) if len(np.unique(true_reasoning)) > 1 else 0.5
        )
        metrics["reasoning_mean_prob"] = float(np.mean(prob_reasoning))
        if np.any(true_reasoning == 1):
            metrics["reasoning_mean_prob_pos"] = float(np.mean(prob_reasoning[true_reasoning == 1]))
        else:
            metrics["reasoning_mean_prob_pos"] = 0.0
        if np.any(true_reasoning == 0):
            metrics["reasoning_mean_prob_neg"] = float(np.mean(prob_reasoning[true_reasoning == 0]))
        else:
            metrics["reasoning_mean_prob_neg"] = 0.0

        if eval_search:
            metrics.update(_classification_metrics(true_search, pred_search, prefix="search_"))
            metrics["search_auc_roc"] = (
                roc_auc_score(true_search, prob_search) if len(np.unique(true_search)) > 1 else 0.5
            )
            metrics["search_mean_prob"] = float(np.mean(prob_search))
            if np.any(true_search == 1):
                metrics["search_mean_prob_pos"] = float(np.mean(prob_search[true_search == 1]))
            else:
                metrics["search_mean_prob_pos"] = 0.0
            if np.any(true_search == 0):
                metrics["search_mean_prob_neg"] = float(np.mean(prob_search[true_search == 0]))
            else:
                metrics["search_mean_prob_neg"] = 0.0
            metrics["combined_accuracy"] = (metrics["reasoning_accuracy"] + metrics["search_accuracy"]) / 2
            metrics["combined_f1"] = (metrics["reasoning_f1"] + metrics["search_f1"]) / 2
            metrics["combined_auc_roc"] = (metrics["reasoning_auc_roc"] + metrics["search_auc_roc"]) / 2
        return metrics

    return compute_metrics


def _make_trainer_cls():
    from transformers import Trainer

    class SFTTrainerWithMetrics(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tok = self.processing_class
            self.zero_id = tok.encode("0", add_special_tokens=False)[0]
            self.one_id = tok.encode("1", add_special_tokens=False)[0]

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)
            reasoning_labels = inputs["reasoning_label"].to(device)
            search_labels = inputs["search_label"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = outputs.loss

                pr, ps, prr, prs = [], [], [], []
                for i in range(labels.shape[0]):
                    non_masked = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    if len(non_masked) == 0:
                        pr.append(0); ps.append(0); prr.append(0.5); prs.append(0.5)
                        continue
                    first = non_masked[0].item()
                    pos = first - 1 if first > 0 else 0
                    r_logits = logits[i, pos, [self.zero_id, self.one_id]]
                    pr.append(r_logits.argmax().item())
                    prr.append(torch.softmax(r_logits, dim=0)[1].item())
                    if pos + 2 < logits.shape[1]:
                        s_logits = logits[i, pos + 2, [self.zero_id, self.one_id]]
                        ps.append(s_logits.argmax().item())
                        prs.append(torch.softmax(s_logits, dim=0)[1].item())
                    else:
                        ps.append(0); prs.append(0.5)

                preds = torch.tensor([pr, ps, prr, prs], dtype=torch.float, device=device).T
                label_tensor = torch.stack([reasoning_labels, search_labels], dim=1).float()

            if prediction_loss_only:
                return (loss, None, None)
            return (loss, preds, label_tensor)

    return SFTTrainerWithMetrics


def _is_main_process() -> bool:
    return int(os.environ["LOCAL_RANK"]) == 0


def _maybe_login_wandb(cfg: SFTConfig, config_dict: dict):
    if not cfg.wandb_enabled or not _is_main_process():
        return None
    try:
        import wandb
    except ImportError:
        print("[sft] wandb not installed; skipping logging.", file=sys.stderr)
        return None
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    elif not os.environ.get("WANDB_MODE") and not os.path.exists(os.path.expanduser("~/.netrc")):
        print("[sft] WANDB_API_KEY not set and no ~/.netrc; wandb will run offline.", file=sys.stderr)
        os.environ.setdefault("WANDB_MODE", "offline")
    return wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=cfg.run_name,
        config=config_dict,
    )


def run(cfg: SFTConfig) -> str:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments

    run_name = cfg.derive_run_name()
    output_dir = os.path.abspath(os.path.join(cfg.output_root, run_name))
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "config.json")
    if _is_main_process():
        with open(config_path, "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)
        print(f"[sft] Run: {run_name}")
        print(f"[sft] Output dir: {output_dir}")
        print(f"[sft] Config saved: {config_path}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and _is_main_process():
        print("[sft] HF_TOKEN not set; gated datasets/models will fail to load.", file=sys.stderr)

    wandb_run = _maybe_login_wandb(cfg, asdict(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model_backbone_path, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if _is_main_process():
        print(f"[sft] Loading dataset: {cfg.dataset}")
    hf_dataset = load_dataset(cfg.dataset, token=hf_token)

    train_split = hf_dataset["train"]
    if cfg.train_sample_filter is not None:
        allowed = set(cfg.train_sample_filter)

        def _filter(example):
            try:
                sample_num = int(example["id"].split("_")[1])
            except (KeyError, IndexError, ValueError):
                return False
            return sample_num in allowed

        train_split = train_split.filter(_filter)
        if _is_main_process():
            print(f"[sft] Filter samples {sorted(allowed)} -> {len(train_split)} examples")

    if cfg.train_subset_size:
        train_split = train_split.shuffle(seed=42).select(range(min(cfg.train_subset_size, len(train_split))))

    train_ds = MedPraDataset(
        train_split, tokenizer,
        max_length=cfg.max_len,
        docs_mode=cfg.docs_mode,
        no_docs_ratio=cfg.no_docs_ratio,
        use_is_correct_as_reasoning=cfg.use_is_correct_as_reasoning,
    )
    dev_split = hf_dataset["dev"]
    if cfg.eval_dev_size:
        dev_split = dev_split.shuffle(seed=42).select(range(min(cfg.eval_dev_size, len(dev_split))))
    eval_ds = MedPraDataset(dev_split, tokenizer, max_length=cfg.max_len, docs_mode="all_docs")

    num_gpus = max(1, torch.cuda.device_count())
    effective_batch = cfg.batch_size * num_gpus * cfg.gradient_accumulation_steps
    total_steps = (len(train_ds) // max(effective_batch, 1)) * cfg.num_epochs
    if _is_main_process():
        print(
            f"[sft] Train={len(train_ds)} Eval={len(eval_ds)} "
            f"EffBatch={effective_batch} EstSteps={total_steps}"
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=50,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        do_train=True,
        do_eval=True,
        fp16=False,
        bf16=True,
        report_to=["wandb"] if wandb_run is not None else [],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        save_only_model=cfg.save_only_model,
    )

    model = AutoModelForCausalLM.from_pretrained(cfg.reward_model_backbone_path, torch_dtype=torch.bfloat16, token=hf_token)
    if _is_main_process():
        print(f"[sft] GPUs: {num_gpus}; model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    class EpochSaveCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
            if not state.is_world_process_zero:
                return
            epoch = int(state.epoch)
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            print(f"[sft] Saving epoch {epoch} to {epoch_dir}")
            model.save_pretrained(epoch_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(epoch_dir)

    TrainerCls = _make_trainer_cls()
    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=_PaddedCollator(tokenizer),
        compute_metrics=_build_compute_metrics(cfg.eval_search_labels),
        callbacks=[EpochSaveCallback()],
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)
    if _is_main_process():
        tokenizer.save_pretrained(final_dir)
        print(f"[sft] Saved final model -> {final_dir}")
    if wandb_run is not None:
        wandb_run.finish()
    return final_dir


def parse_args(argv=None) -> SFTConfig:
    p = argparse.ArgumentParser(description="PRA SFT trainer")
    p.add_argument("--config", type=str, required=True, help="Path to YAML SFTConfig")
    p.add_argument("--reward_model_backbone_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--output_root", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    ns = p.parse_args(argv)
    overrides = {k: v for k, v in vars(ns).items() if k != "config" and v is not None}
    return load_config(ns.config, overrides=overrides)


def main_cli() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main_cli()
