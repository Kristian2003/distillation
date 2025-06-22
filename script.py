import argparse
import json
import os
import time
import re
from contextlib import ExitStack
import gzip
import zipfile
import io
from difflib import SequenceMatcher
from typing import Optional, Any, TypedDict, Iterator, List, Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import get_worker_info
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    AutoModelForSeq2SeqLM
)
from tqdm import tqdm
import numpy as np

try:
    from compress import decompress_logits
except ImportError:
    def decompress_logits(compressed_data: Any) -> Optional[np.ndarray]:
        print("WARNING: 'compress.decompress_logits' not found. Returning None for teacher_logits.")
        print("Ensure 'compress.py' is in the same directory or 'compress' module is installed.")
        return None

class RawDistillEntry(TypedDict):
    focal_file: Optional[str]
    test_method_masked: str
    original_target: Optional[str | List[str]]
    assertions: Optional[List[str]]
    predicted_assertions: Optional[str]
    model_prediction: Optional[str]
    compressed_logits: Dict[str, Any]

class ProcessedDistillSample(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    hard_labels: torch.Tensor
    teacher_logits: Optional[torch.Tensor]
    ground_truth_text: str
    teacher_predicted_text: str
    original_input_text: str

def normalize_assertion_script1(assertion: str) -> str:
    assertion = re.sub(r'\s+', ' ', assertion).strip()
    assertion = re.sub(r'assertEquals\s*\(\s*[^,]+,\s*([^)]+)\s*\)', r'assertEquals(VALUE, \1)', assertion)
    assertion = re.sub(r'assert(Equals|That|True|False|Null|NotNull|Throws|Same|NotSame|ArrayEquals)\b', r'assert\1', assertion, flags=re.IGNORECASE)
    return assertion

def calculate_similarity_script1(reference: str, candidate: str) -> float:
    if not isinstance(reference, str) or not isinstance(candidate, str): return 0.0
    if not reference or not candidate: return 0.0
    return SequenceMatcher(None, reference, candidate).ratio()

def classify_assertion_type_script1(assertion: str) -> str:
    if not isinstance(assertion, str): return "other"
    assertion_lower = assertion.lower()
    if re.search(r"\bassertequals\b", assertion_lower) or \
       (re.search(r"\bassertthat\b", assertion_lower) and ".isequalto" in assertion_lower): return "equality"
    elif re.search(r"\basserttrue\b", assertion_lower): return "truth"
    elif re.search(r"\bassertfalse\b", assertion_lower): return "falsity"
    elif re.search(r"\bassertnull\b", assertion_lower): return "null"
    elif re.search(r"\bassertnotnull\b", assertion_lower): return "not_null"
    elif re.search(r"\bassertthrows\b", assertion_lower): return "exception"
    elif re.search(r"\bassertsame\b", assertion_lower): return "same"
    elif re.search(r"\bassertnotsame\b", assertion_lower): return "not_same"
    elif re.search(r"\bassertarrayequals\b", assertion_lower): return "array_equality"
    else: return "other"

def evaluate_assertions_script1_style(generated_assertions_block: str, reference_assertions_block: str) -> Dict[str, Any]:
    def _parse_assertions(block: Any) -> List[str]:
        if not block or not isinstance(block, str): return []
        items = re.split(r';|\n', block)
        return [a.strip() + (';' if a.strip() and not a.strip().endswith(';') else '') for a in items if a.strip()]

    generated_list_raw = _parse_assertions(generated_assertions_block)
    reference_list_raw = _parse_assertions(reference_assertions_block)
    normalized_generated = [normalize_assertion_script1(a) for a in generated_list_raw]
    normalized_reference = [normalize_assertion_script1(a) for a in reference_list_raw]
    exact_matches = 0
    for gen_item in normalized_generated:
        if gen_item in normalized_reference: exact_matches +=1
    similarity_scores_list = []
    if normalized_generated:
        for gen_norm in normalized_generated:
            best_sim = 0.0
            if normalized_reference:
                for ref_norm in normalized_reference:
                    sim = calculate_similarity_script1(gen_norm, ref_norm)
                    best_sim = max(best_sim, sim)
            similarity_scores_list.append(best_sim)
    gen_types = [classify_assertion_type_script1(a) for a in generated_list_raw]
    ref_types = [classify_assertion_type_script1(a) for a in reference_list_raw]
    gen_type_counts = {t: gen_types.count(t) for t in set(gen_types)}
    ref_type_counts = {t: ref_types.count(t) for t in set(ref_types)}

    if not normalized_generated and not normalized_reference:
        precision, recall, f1, accuracy_script1 = 1.0, 1.0, 1.0, 1.0
        similarity_avg_script1 = 1.0
    else:
        precision = exact_matches / len(normalized_generated) if normalized_generated else 0.0
        recall = exact_matches / len(normalized_reference) if normalized_reference else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        max_len = max(len(normalized_generated), len(normalized_reference))
        accuracy_script1 = exact_matches / max_len if max_len > 0 else 0.0
        similarity_avg_script1 = sum(similarity_scores_list) / len(similarity_scores_list) if similarity_scores_list else 0.0
        if not normalized_generated and normalized_reference: similarity_avg_script1 = 0.0
    return {"exact_matches_script1": exact_matches, "generated_count_script1": len(normalized_generated),
            "reference_count_script1": len(normalized_reference), "precision_script1": precision, "recall_script1": recall,
            "f1_script1": f1, "accuracy_script1": accuracy_script1, "similarity_avg_script1": similarity_avg_script1,
            "gen_type_counts_script1": gen_type_counts, "ref_type_counts_script1": ref_type_counts,
            "similarity_scores_list_script1": similarity_scores_list}

def get_model_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 * 1024)

def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    return obj

class NestedZipDistillationDataset(IterableDataset):
    def __init__(self, data_path: str, student_tokenizer: T5Tokenizer,
                 student_max_src_length: int, student_max_tgt_length: int,
                 outer_zip_internal_path_to_inner_zip: str,
                 inner_zip_internal_filename_jsonl: str,
                 jsonl_inside_inner_zip_is_gzipped: bool = False,
                 focal_content_field_name: str = "focal_file"):
        super().__init__()
        self.data_path = data_path
        self.student_tokenizer = student_tokenizer
        self.student_max_src_length = student_max_src_length
        self.student_max_tgt_length = student_max_tgt_length
        self.outer_zip_internal_path_to_inner_zip = outer_zip_internal_path_to_inner_zip
        self.inner_zip_internal_filename_jsonl = inner_zip_internal_filename_jsonl
        self.jsonl_inside_inner_zip_is_gzipped = jsonl_inside_inner_zip_is_gzipped
        self.focal_content_field_name = focal_content_field_name
        self._num_items = 0
        self._calculate_num_items()

    def _open_final_jsonl_stream(self, exit_stack: ExitStack) -> io.TextIOWrapper:
        try:
            outer_archive = exit_stack.enter_context(zipfile.ZipFile(self.data_path, 'r'))
            try: inner_zip_bytes = outer_archive.read(self.outer_zip_internal_path_to_inner_zip)
            except KeyError: raise FileNotFoundError(f"Inner zip '{self.outer_zip_internal_path_to_inner_zip}' not found in outer zip '{self.data_path}'. Available: {outer_archive.namelist()}")
            inner_zip_buffer = io.BytesIO(inner_zip_bytes)
            inner_archive = exit_stack.enter_context(zipfile.ZipFile(inner_zip_buffer, 'r'))
            try:
                jsonl_binary_stream = exit_stack.enter_context(inner_archive.open(self.inner_zip_internal_filename_jsonl, 'r'))
                if self.jsonl_inside_inner_zip_is_gzipped:
                    gzipped_file = exit_stack.enter_context(gzip.GzipFile(fileobj=jsonl_binary_stream))
                    text_stream = io.TextIOWrapper(gzipped_file, encoding='utf-8')
                else: text_stream = io.TextIOWrapper(jsonl_binary_stream, encoding='utf-8')
                return text_stream
            except KeyError: raise FileNotFoundError(f"JSONL file '{self.inner_zip_internal_filename_jsonl}' not found in inner zip. Available: {zipfile.ZipFile(io.BytesIO(inner_zip_bytes), 'r').namelist()}")
        except zipfile.BadZipFile as e: raise ValueError(f"Bad zip file: {e}")
        except Exception as e: raise RuntimeError(f"Unexpected error opening stream: {e}")

    def _is_valid_entry_dict(self, entry_dict: RawDistillEntry) -> bool:
        has_target = 'original_target' in entry_dict or 'assertions' in entry_dict
        has_teacher_pred_text = 'model_prediction' in entry_dict or 'predicted_assertions' in entry_dict
        has_logits = 'compressed_logits' in entry_dict and entry_dict['compressed_logits'] is not None
        return (isinstance(entry_dict.get('test_method_masked'), str) and has_target and has_teacher_pred_text and has_logits)

    def _calculate_num_items(self):
        count = 0
        print(f"Calculating number of items in {self.data_path} for {self.outer_zip_internal_path_to_inner_zip} -> {self.inner_zip_internal_filename_jsonl}...")
        try:
            with ExitStack() as stack:
                text_stream = self._open_final_jsonl_stream(stack)
                first_line_skipped = False
                for line in text_stream:
                    line_content = line.strip()
                    if not line_content: continue
                    try:
                        entry_data = json.loads(line_content)
                        if not first_line_skipped and isinstance(entry_data, dict) and 'header' in entry_data:
                            first_line_skipped = True; continue
                        first_line_skipped = True
                        entry = convert_numpy_types(entry_data)
                        if self._is_valid_entry_dict(entry):
                            count += 1
                    except json.JSONDecodeError: continue
            self._num_items = count
            print(f"Found {self._num_items} valid items.")
        except Exception as e:
            print(f"Error during _calculate_num_items: {e}. Num items set to 0."); self._num_items = 0

    def _process_entry(self, entry_data: Dict[str, Any]) -> Optional[ProcessedDistillSample]:
        entry: RawDistillEntry = convert_numpy_types(entry_data)

        if not self._is_valid_entry_dict(entry):
            return None

        focal_code_str = entry.get(self.focal_content_field_name, "")
        if not isinstance(focal_code_str, str): focal_code_str = ""
        test_method_str = entry['test_method_masked']
        test_method_prefix = "TEST METHOD:\n"
        
        if focal_code_str.strip():
            test_method_full_text = f"{test_method_prefix}{test_method_str}"
            test_method_tokens_check = self.student_tokenizer(test_method_full_text, add_special_tokens=True, return_attention_mask=False)
            test_method_len_no_trunc = len(test_method_tokens_check.input_ids)
            focal_code_prefix = "FOCAL CODE:\n"
            overhead_text = f"{focal_code_prefix}\n\n{test_method_prefix}"
            overhead_tokens = self.student_tokenizer(overhead_text, add_special_tokens=False).input_ids
            estimated_overhead_len = len(overhead_tokens) + 1

            if test_method_len_no_trunc >= self.student_max_src_length - estimated_overhead_len - 10:
                input_text = test_method_full_text
            else:
                space_for_focal_text_tokens = self.student_max_src_length - test_method_len_no_trunc - estimated_overhead_len
                if space_for_focal_text_tokens <= 0:
                    input_text = test_method_full_text
                else:
                    focal_tokens = self.student_tokenizer(focal_code_str,add_special_tokens=False, max_length=space_for_focal_text_tokens, truncation=True, return_attention_mask=False)
                    truncated_focal_str = self.student_tokenizer.decode(focal_tokens.input_ids, skip_special_tokens=True)
                    input_text = f"{focal_code_prefix}{truncated_focal_str}\n\n{test_method_full_text}"
        else:
            input_text = f"{test_method_prefix}{test_method_str}"

        source_encoding = self.student_tokenizer(input_text, max_length=self.student_max_src_length, padding="max_length", truncation=True, return_tensors="pt")

        if 'original_target' in entry and entry['original_target'] is not None:
            gt_text_input = entry['original_target']
            ground_truth_assertions_text = "\n".join(gt_text_input) if isinstance(gt_text_input, list) else str(gt_text_input)
        elif 'assertions' in entry and entry['assertions'] is not None:
            ground_truth_assertions_text = "\n".join(str(s) for s in entry['assertions'])
        else: return None

        if 'model_prediction' in entry and entry['model_prediction'] is not None:
            teacher_predicted_text = str(entry['model_prediction'])
        elif 'predicted_assertions' in entry and entry['predicted_assertions'] is not None:
            teacher_predicted_text = str(entry['predicted_assertions'])
        else: return None

        target_encoding = self.student_tokenizer(ground_truth_assertions_text, max_length=self.student_max_tgt_length, padding="max_length", truncation=True, return_tensors="pt")
        hard_labels = target_encoding["input_ids"].squeeze(0)
        hard_labels[hard_labels == self.student_tokenizer.pad_token_id] = -100

        teacher_logits_decompressed_np = decompress_logits(entry['compressed_logits'])
        final_teacher_logits = None
        if teacher_logits_decompressed_np is not None:
            final_teacher_logits = torch.tensor(teacher_logits_decompressed_np, dtype=torch.float32)
            if final_teacher_logits.ndim == 2: final_teacher_logits = final_teacher_logits.unsqueeze(0)
            elif final_teacher_logits.ndim == 3:
                if final_teacher_logits.shape[0] != 1: final_teacher_logits = final_teacher_logits[0:1, :, :]
            else: final_teacher_logits = None
            if final_teacher_logits is not None: final_teacher_logits = final_teacher_logits.squeeze(0)

        return {"input_ids": source_encoding["input_ids"].squeeze(0), "attention_mask": source_encoding["attention_mask"].squeeze(0),
                "hard_labels": hard_labels, "teacher_logits": final_teacher_logits, "ground_truth_text": ground_truth_assertions_text,
                "teacher_predicted_text": teacher_predicted_text, "original_input_text": input_text}

    def __iter__(self) -> Iterator[ProcessedDistillSample]:
        worker_info = get_worker_info()
        with ExitStack() as stack:
            try: file_stream_for_iter = self._open_final_jsonl_stream(stack)
            except Exception as e: print(f"Worker {worker_info.id if worker_info else 'main'} failed to open stream: {e}"); return
            first_line_skipped = False; item_idx = 0
            for line_content in file_stream_for_iter:
                line_strip = line_content.strip()
                if not line_strip: continue
                if worker_info is None or (item_idx % worker_info.num_workers == worker_info.id):
                    try:
                        entry_data = json.loads(line_strip)
                        if not first_line_skipped and isinstance(entry_data, dict) and 'header' in entry_data:
                            first_line_skipped = True; continue
                        first_line_skipped = True
                        processed_item = self._process_entry(entry_data)
                        if processed_item: yield processed_item
                    except json.JSONDecodeError: pass
                    except Exception: pass
                item_idx += 1
    def __len__(self) -> int: return self._num_items

class DistillationLoss(torch.nn.Module):
    def __init__(self, temperature: float, alpha: float):
        super().__init__()
        self.temperature = temperature
        self.alpha_kl = alpha

    def forward(self, student_logits: torch.Tensor, teacher_logits: Optional[torch.Tensor],
                hard_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        student_vocab_size = student_logits.size(-1)
        loss_h = F.cross_entropy(student_logits.reshape(-1, student_vocab_size), hard_labels.reshape(-1), ignore_index=-100)
        loss_s = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)

        if teacher_logits is not None and self.alpha_kl > 0:
            teacher_vocab_size = teacher_logits.size(-1)
            batch_size = student_logits.size(0)
            student_seq_len = student_logits.size(1)
            teacher_seq_len = teacher_logits.size(1)
            min_seq_len = min(student_seq_len, teacher_seq_len)
            current_teacher_logits = None

            if teacher_logits.ndim == 2 and batch_size == 1: current_teacher_logits = teacher_logits.unsqueeze(0)[:, :min_seq_len, :]
            elif teacher_logits.ndim == 3 and teacher_logits.size(0) == batch_size: current_teacher_logits = teacher_logits[:, :min_seq_len, :]
            else: print(f"Warning: KLDiv: Mismatch teacher_logits. Student: {student_logits.shape}, Teacher: {teacher_logits.shape}. Skipping KL.")

            if current_teacher_logits is not None:
                current_student_logits_for_kl = student_logits[:, :min_seq_len, :]
                current_hard_labels_for_kl_mask = hard_labels[:, :min_seq_len]
                min_vocab_size = min(student_vocab_size, teacher_vocab_size)
                student_log_probs = F.log_softmax(current_student_logits_for_kl[..., :min_vocab_size] / self.temperature, dim=-1)
                teacher_probs = F.softmax(current_teacher_logits[..., :min_vocab_size] / self.temperature, dim=-1).detach()
                kl_div_element_wise = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
                mask = (current_hard_labels_for_kl_mask != -100).float()
                if kl_div_element_wise.shape != mask.shape: raise ValueError(f"Shape mismatch KLDiv elements {kl_div_element_wise.shape} and mask {mask.shape}")
                loss_s_masked_sum = (kl_div_element_wise * mask).sum()
                num_valid_tokens_for_kl = mask.sum().clamp(min=1e-9)
                loss_s = loss_s_masked_sum / num_valid_tokens_for_kl
                loss_s = loss_s * (self.temperature ** 2)
        total_loss = (1.0 - self.alpha_kl) * loss_h + self.alpha_kl * loss_s
        return total_loss, loss_h, loss_s

def evaluate_student(model: T5ForConditionalGeneration, tokenizer: T5Tokenizer,
                     dataloader: DataLoader, device: torch.device,
                     criterion: DistillationLoss, args: argparse.Namespace):
    model.eval()
    total_eval_loss, total_hard_loss, total_soft_loss = 0.0, 0.0, 0.0
    num_eval_batches = 0; num_samples_processed = 0; inference_times_ms = []
    all_metrics_s_vs_t_script1: List[Dict] = []; all_metrics_s_vs_gt_script1: List[Dict] = []

    try: dataloader_len = len(dataloader)
    except TypeError: dataloader_len = None

    default_metrics = { "avg_loss": 0.0, "avg_hard_loss": 0.0, "avg_soft_loss": 0.0,
                       **{f"avg_{m}_s_vs_t_script1": 0.0 for m in ["precision", "recall", "f1", "accuracy", "similarity_avg"]},
                       **{f"avg_{m}_s_vs_gt_script1": 0.0 for m in ["precision", "recall", "f1", "accuracy", "similarity_avg"]},
                       "avg_inference_time_per_sample_ms": 0.0, "num_eval_samples": 0 }
    if dataloader_len is not None and dataloader_len == 0:
        print("Warning: Eval dataloader empty."); return default_metrics

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating", total=dataloader_len, disable=dataloader_len is None)):
            batch: ProcessedDistillSample = batch_data
            input_ids = batch["input_ids"].to(device); attention_mask = batch["attention_mask"].to(device)
            hard_labels = batch["hard_labels"].to(device)
            raw_teacher_logits_batch = batch.get("teacher_logits")
            teacher_logits_for_loss = None
            if raw_teacher_logits_batch is not None:
                 teacher_logits_for_loss = raw_teacher_logits_batch.to(device)
            ground_truth_texts: List[str] = batch["ground_truth_text"]
            teacher_predicted_texts: List[str] = batch["teacher_predicted_text"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=hard_labels)
            student_logits = outputs.logits
            loss, loss_h, loss_s = criterion(student_logits, teacher_logits_for_loss, hard_labels)
            total_eval_loss += loss.item(); total_hard_loss += loss_h.item(); total_soft_loss += loss_s.item()
            num_eval_batches +=1
            start_time = time.perf_counter()
            generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=args.student_max_tgt_length, num_beams=args.num_beams, early_stopping=True)
            inference_times_ms.append((time.perf_counter() - start_time) * 1000 / input_ids.size(0) if input_ids.size(0) > 0 else 0)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for i in range(len(generated_texts)):
                gen_text, teacher_text, gt_text = generated_texts[i], teacher_predicted_texts[i], ground_truth_texts[i]
                all_metrics_s_vs_t_script1.append(evaluate_assertions_script1_style(gen_text, teacher_text))
                all_metrics_s_vs_gt_script1.append(evaluate_assertions_script1_style(gen_text, gt_text))
            num_samples_processed += input_ids.size(0)
            if batch_idx == 0 and num_samples_processed > 0:
                print(f"\nGT: {ground_truth_texts[0][:100]}...\nTeacher: {teacher_predicted_texts[0][:100]}...\nStudent: {generated_texts[0][:100]}...")
                if all_metrics_s_vs_t_script1: print(f"  S1(S|T): Acc={all_metrics_s_vs_t_script1[0]['accuracy_script1']:.3f} Sim={all_metrics_s_vs_t_script1[0]['similarity_avg_script1']:.3f}")
                if all_metrics_s_vs_gt_script1: print(f"  S1(S|GT): Acc={all_metrics_s_vs_gt_script1[0]['accuracy_script1']:.3f} Sim={all_metrics_s_vs_gt_script1[0]['similarity_avg_script1']:.3f}")
    if num_eval_batches == 0: return default_metrics
    results = {"avg_loss": total_eval_loss/num_eval_batches, "avg_hard_loss":total_hard_loss/num_eval_batches, "avg_soft_loss": total_soft_loss/num_eval_batches}
    metric_keys_script1 = ["precision_script1", "recall_script1", "f1_script1", "accuracy_script1", "similarity_avg_script1"]
    for prefix, metrics_list in [("s_vs_t", all_metrics_s_vs_t_script1), ("s_vs_gt", all_metrics_s_vs_gt_script1)]:
        for key in metric_keys_script1:
            avg_val = np.mean([m[key] for m in metrics_list]) if metrics_list else 0.0
            results[f"avg_{key.replace('_avg_script1', '_avg').replace('_script1', '')}_{prefix}_script1"] = float(avg_val)
    avg_inf_time_per_sample_ms = np.mean(inference_times_ms) if inference_times_ms else 0.0
    
    results.update({"avg_inference_time_per_sample_ms": avg_inf_time_per_sample_ms, "num_eval_samples": num_samples_processed})
    return results

def save_distillation_summary(args: argparse.Namespace, final_eval_results: Dict, student_model_size_mb: float, teacher_model_size_mb: float, epochs_trained: int, training_duration_s: float):
    summary_path = os.path.join(args.output_dir, "distillation_summary.json")
    args_dict = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v for k, v in vars(args).items()}
    summary_data = {"args": args_dict, "epochs_trained": epochs_trained, "training_duration_seconds": round(training_duration_s, 2),
                    "student_model_name": args.student_model_name, "student_model_size_mb": round(student_model_size_mb, 2),
                    "teacher_model_name": args.teacher_model_name, "teacher_model_size_mb": round(teacher_model_size_mb, 2) if teacher_model_size_mb > 0 else "N/A",
                    "compression_ratio_model_size": (round(student_model_size_mb / teacher_model_size_mb, 4)) if teacher_model_size_mb > 0 and student_model_size_mb > 0 else "N/A",
                    "final_evaluation_metrics": convert_numpy_types(final_eval_results)}
    try:
        with open(summary_path, "w") as f: json.dump(summary_data, f, indent=2)
        print(f"Distillation summary saved to {summary_path}")
    except Exception as e: print(f"Error saving summary: {e}")

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation (Simplified Student Config)")

    parser.add_argument("--distillation_data_path", type=str, required=True)
    parser.add_argument("--train_outer_zip_internal_path", type=str, required=True)
    parser.add_argument("--train_inner_zip_jsonl_filename", type=str, required=True)
    parser.add_argument("--train_jsonl_is_gzipped", action="store_true")
    parser.add_argument("--val_outer_zip_internal_path", type=str, required=True)
    parser.add_argument("--val_inner_zip_jsonl_filename", type=str, required=True)
    parser.add_argument("--val_jsonl_is_gzipped", action="store_true")
    parser.add_argument("--focal_content_field_name", type=str, default="focal_file")
    
    parser.add_argument("--teacher_model_name", type=str, default="Salesforce/codet5-base")
    parser.add_argument("--student_model_name", type=str, default="Salesforce/codet5-small")
    parser.add_argument("--student_max_src_length", type=int, default=1024)
    parser.add_argument("--student_max_tgt_length", type=int, default=512)
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps_ratio", type=float, default=0.1)
    parser.add_argument("--alpha_kl_loss_weight", type=float, default=0.7,
                        help="Weight for SOFT KL loss (1-alpha for hard CE loss). Template default was 0.7 for KL.")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--save_best_model", action="store_true")
    parser.add_argument("--best_model_metric", type=str, default="avg_f1_s_vs_gt_script1",
                        choices=["avg_f1_s_vs_t_script1", "avg_accuracy_s_vs_t_script1", "avg_similarity_avg_s_vs_t_script1",
                                  "avg_f1_s_vs_gt_script1", "avg_accuracy_s_vs_gt_script1", "avg_similarity_avg_s_vs_gt_script1",
                                  "avg_loss"],
                        help="Metric for saving best model. Lower for 'avg_loss'.")
    parser.add_argument("--num_dataloader_workers", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()

    start_train_time = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading student tokenizer: {args.student_model_name}")
    try:
        student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name)
    except Exception:
        print(f"AutoTokenizer failed for {args.student_model_name}, trying T5Tokenizer.")
        student_tokenizer = T5Tokenizer.from_pretrained(args.student_model_name)

    print(f"Loading student model with standard config: {args.student_model_name}")
    student_model = T5ForConditionalGeneration.from_pretrained(args.student_model_name).to(device)

    student_vocab_size = student_model.config.vocab_size
    student_model_size_mb = get_model_size_mb(student_model)

    teacher_vocab_size = -1
    teacher_model_size_mb = -1.0
    print(f"Attempting to load teacher model ({args.teacher_model_name}) for metadata...")
    try:
        _teacher_config = AutoConfig.from_pretrained(args.teacher_model_name)
        teacher_vocab_size = _teacher_config.vocab_size
        print(f"Teacher model ('{args.teacher_model_name}') metadata: Vocab {teacher_vocab_size}")
    except Exception as e:
        print(f"Warning: Could not load teacher model config '{args.teacher_model_name}' for metadata: {e}. Defaulting teacher vocab to student's.")
        teacher_vocab_size = student_vocab_size

    common_dataset_args = { "student_tokenizer": student_tokenizer, "student_max_src_length": args.student_max_src_length,
                           "student_max_tgt_length": args.student_max_tgt_length, "focal_content_field_name": args.focal_content_field_name}
    train_dataset = NestedZipDistillationDataset(data_path=args.distillation_data_path, outer_zip_internal_path_to_inner_zip=args.train_outer_zip_internal_path,
                                                 inner_zip_internal_filename_jsonl=args.train_inner_zip_jsonl_filename, jsonl_inside_inner_zip_is_gzipped=args.train_jsonl_is_gzipped, **common_dataset_args)
    val_dataset = NestedZipDistillationDataset(data_path=args.distillation_data_path, outer_zip_internal_path_to_inner_zip=args.val_outer_zip_internal_path,
                                               inner_zip_internal_filename_jsonl=args.val_inner_zip_jsonl_filename, jsonl_inside_inner_zip_is_gzipped=args.val_jsonl_is_gzipped, **common_dataset_args)

    if len(train_dataset) == 0: print("Error: Training dataset empty. Exiting."); return
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, pin_memory=device.type=='cuda', drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=args.num_dataloader_workers, pin_memory=device.type=='cuda')

    optimizer = AdamW(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_train_samples = len(train_dataset)
    if num_train_samples > 0 and args.batch_size > 0 and args.gradient_accumulation_steps > 0:
        num_update_steps_per_epoch = max(1, num_train_samples // (args.batch_size * args.gradient_accumulation_steps))
        total_training_steps = num_update_steps_per_epoch * args.epochs
    else: total_training_steps = (1000 // args.gradient_accumulation_steps) * args.epochs; num_update_steps_per_epoch = total_training_steps // args.epochs
    warmup_steps = int(args.warmup_steps_ratio * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    distillation_criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha_kl_loss_weight).to(device)

    best_val_metric_score = -float('inf') if args.best_model_metric != "avg_loss" else float('inf')
    global_step = 0; epochs_completed = 0
    print(f"\n--- Starting Distillation Training ({args.epochs} epochs) ---")
    print(f"  Student: {args.student_model_name} ({student_model_size_mb:.2f} MB), Vocab: {student_vocab_size}")
    if teacher_vocab_size != -1 : print(f"  Teacher: {args.teacher_model_name}, Vocab: {teacher_vocab_size}")
    if student_vocab_size != teacher_vocab_size and teacher_vocab_size != -1: print(f"  Warning: Vocab sizes differ (S:{student_vocab_size}, T:{teacher_vocab_size}). KLDiv uses min_vocab_size.")
    print(f"  Best model by: {args.best_model_metric} ({'higher' if args.best_model_metric != 'avg_loss' else 'lower'} is better)")
    print(f"  Alpha (KL Loss Weight): {args.alpha_kl_loss_weight}")

    for epoch in range(args.epochs):
        student_model.train()
        epoch_train_loss, epoch_hard_loss, epoch_soft_loss = 0.0, 0.0, 0.0
        num_batches_processed_in_epoch = 0
        try: total_batches_for_pbar = len(train_dataloader)
        except TypeError: total_batches_for_pbar = None
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", total=total_batches_for_pbar, disable=total_batches_for_pbar is None)
        optimizer.zero_grad()
        for batch_idx, batch_data in enumerate(progress_bar):
            batch: ProcessedDistillSample = batch_data
            input_ids = batch["input_ids"].to(device); attention_mask = batch["attention_mask"].to(device)
            hard_labels = batch["hard_labels"].to(device)
            raw_teacher_logits = batch.get("teacher_logits")
            teacher_logits_for_loss: Optional[torch.Tensor] = None
            if raw_teacher_logits is not None and isinstance(raw_teacher_logits, torch.Tensor):
                teacher_logits_for_loss = raw_teacher_logits.to(device)
            outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=hard_labels)
            s_logits = outputs.logits
            loss, loss_h, loss_s = distillation_criterion(s_logits, teacher_logits_for_loss, hard_labels)
            if args.gradient_accumulation_steps > 1: loss = loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_train_loss += loss.item() * args.gradient_accumulation_steps
            epoch_hard_loss += loss_h.item()
            epoch_soft_loss += loss_s.item() if loss_s is not None else 0.0
            num_batches_processed_in_epoch +=1
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == total_batches_for_pbar:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1
            if global_step > 0 and global_step % args.log_steps == 0 and (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                lr = scheduler.get_last_lr()[0]
                avg_loss_so_far = epoch_train_loss/num_batches_processed_in_epoch
                avg_h_loss_so_far = epoch_hard_loss/num_batches_processed_in_epoch
                avg_s_loss_so_far = epoch_soft_loss/num_batches_processed_in_epoch
                progress_bar.set_postfix({"LR": f"{lr:.2e}", "L":f"{avg_loss_so_far:.3f}","hL":f"{avg_h_loss_so_far:.3f}","sL":f"{avg_s_loss_so_far:.3f}"})
        avg_epoch_loss = epoch_train_loss / num_batches_processed_in_epoch if num_batches_processed_in_epoch > 0 else 0
        print(f"Epoch {epoch+1} Train: AvgLoss:{avg_epoch_loss:.4f} (H:{epoch_hard_loss/num_batches_processed_in_epoch:.4f},S:{epoch_soft_loss/num_batches_processed_in_epoch:.4f}) LR: {scheduler.get_last_lr()[0]:.2e}")

        if len(val_dataset) > 0:
            eval_results = evaluate_student(student_model, student_tokenizer, val_dataloader, device, distillation_criterion, args)
            print(f"  Epoch {epoch+1} Eval: Loss:{eval_results.get('avg_loss',0):.4f} (H:{eval_results.get('avg_hard_loss',0):.4f} S:{eval_results.get('avg_soft_loss',0):.4f})")
            print(f"    S1 Metrics (S vs T):  Acc_S1={eval_results.get('avg_accuracy_s_vs_t_script1',0):.4f}, Sim_S1={eval_results.get('avg_similarity_avg_s_vs_t_script1',0):.4f}, F1_S1={eval_results.get('avg_f1_s_vs_t_script1',0):.4f}")
            print(f"    S1 Metrics (S vs GT): Acc_S1={eval_results.get('avg_accuracy_s_vs_gt_script1',0):.4f}, Sim_S1={eval_results.get('avg_similarity_avg_s_vs_gt_script1',0):.4f}, F1_S1={eval_results.get('avg_f1_s_vs_gt_script1',0):.4f}")
            current_metric_value = eval_results.get(args.best_model_metric, None)
            if current_metric_value is not None:
                save_cond = (args.best_model_metric == "avg_loss" and current_metric_value < best_val_metric_score) or \
                            (args.best_model_metric != "avg_loss" and current_metric_value > best_val_metric_score)
                if args.save_best_model and save_cond:
                    best_val_metric_score = current_metric_value; print(f"    New best {args.best_model_metric}: {best_val_metric_score:.4f}. Saving...")
                    best_model_path = os.path.join(args.output_dir, "best_student_model")
                    student_model.save_pretrained(best_model_path); student_tokenizer.save_pretrained(best_model_path)
                    with open(os.path.join(best_model_path, "eval_results_best.json"),"w") as f: json.dump(convert_numpy_types(eval_results),f,indent=2)
                    with open(os.path.join(best_model_path, "training_args_best.json"),"w") as f: json.dump(vars(args),f,indent=2)
        epochs_completed = epoch + 1
    
    print("Saving final student model..."); final_model_path = os.path.join(args.output_dir, "final_student_model")
    student_model.save_pretrained(final_model_path); student_tokenizer.save_pretrained(final_model_path)
    with open(os.path.join(final_model_path, "training_args_final.json"),"w") as f: json.dump(vars(args),f,indent=2)
    training_duration_s = time.time() - start_train_time
    print(f"\n--- Distillation Complete ({training_duration_s:.2f}s) ---")
    print(f"Student Model ({args.student_model_name}): Size {student_model_size_mb:.2f} MB")
    if teacher_model_size_mb > 0: print(f"Teacher Model ({args.teacher_model_name}): Size {teacher_model_size_mb:.2f} MB")

    final_eval_results = {}
    if len(val_dataset) > 0:
        print("Running final evaluation..."); final_eval_results = evaluate_student(student_model, student_tokenizer, val_dataloader, device, distillation_criterion, args)
        print(f"Final Eval: Loss:{final_eval_results.get('avg_loss',0):.4f} Acc(S|GT):{final_eval_results.get('avg_accuracy_s_vs_gt_script1',0):.4f}")
    save_distillation_summary(args, final_eval_results, student_model_size_mb, teacher_model_size_mb, epochs_completed, training_duration_s)
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()