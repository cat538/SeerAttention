import os
import json
import fcntl
import random
import pickle
import argparse
import datasets
import gc
import sys
import torch
import warnings
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizer
from tqdm import tqdm
from typing import Literal
from seer_attn import SeerAttnLlamaForCausalLM, SeerAttnQwen2ForCausalLM

from proj_config import *

def get_dataset(data_id: str, train_nsamples, seed, seqlen, tokenizer:PreTrainedTokenizer, model_id: str, test_only=False):
    MODEL_ID2CACHE = {
        m: {
            d: {
                t : f"{DATA_DIR}/cache/{m}-{d}-{t}.pkl"
                for t in ["train", "test"]
            } for d in ["wiki", "pg19", "c4"] 
        } for m in ["qwen2.5-3b"]
    }
    if data_id == "pg19": assert test_only

    def lookup_cache(m: str, t: Literal["test", "train"]):
        cache = MODEL_ID2CACHE[model_id][data_id][t]
        print(f"cache: {cache}")
        if not os.path.exists(cache):
            data = datasets.load_dataset(DATA_ID2PATH[data_id], split=t)
            encd = tokenizer("\n\n".join(data['text']), return_tensors='pt')["input_ids"]
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(encd, f)
        else:
            with open(cache, "rb") as f:
                encd = pickle.load(f)
        return encd
    
    print(f"Loading {data_id} tokenized with {model_id} ...")
    testenc = lookup_cache(model_id, "test")
    if test_only:
        return None, testenc

    random.seed(seed)
    trainloader = []
    trainenc = lookup_cache(model_id, "train")
    for _ in range(train_nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    return trainloader, testenc


def append_with_lock(filename: str, data: str):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        fcntl.flock(sys.stdout.fileno(), fcntl.LOCK_EX)
        print(f">>> Writing to {filename} ...", flush=True)
        fcntl.flock(sys.stdout.fileno(), fcntl.LOCK_UN)
        f.write(f"{data}\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@torch.no_grad()
def eval_ppl(
    ppl_testenc: torch.Tensor,
    model,
    input_len=4096
):
    model_use_cache = model.config.use_cache
    model.config.use_cache = False
    nsamples = ppl_testenc.numel() // input_len
    nlls = []

    loss_fct = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(nsamples), desc="Eval ppl", unit="sample"):
        # [bs, input_len]
        batch = ppl_testenc[:, (i * input_len) : ((i + 1) * input_len)].to(model.device)
        outputs = model.model(batch)
        hidden_states = outputs[0]
        # [bs, input_len, vocab_size]
        logits = model.lm_head(hidden_states)
        # [bs, input_len-1, vocab_size]
        shift_logits = logits[:, :-1, :]
        # [bs, input_len-1]
        shift_labels = batch[:, 1:].to(model.lm_head.weight.device)
        loss = loss_fct(
            # [bs * (input_len-1), vocab_size]
            shift_logits.view(-1, shift_logits.size(-1)),
            # [bs * (input_len-1)]
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * input_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * input_len)).item()
    print(f"ppl: {ppl}")
    model.config.use_cache = model_use_cache
    return ppl


def compute_perplexity(
    encodings, 
    model, 
    tokenizer, 
    add_start_token: bool = True, 
    max_length=None, 
    sliding_window=256, 
    truncate=False, 
    hide_progress=False,
):
    device = "cuda"

    if add_start_token:
        assert tokenizer.bos_token is not None, "Input model must have a BOS token"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len

    pbar = tqdm(total=len(encoded_texts), disable=hide_progress)
    nlls = []
    sparsity = []
    
    for encoding_index in range(len(encoded_texts)):
        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        seq_len = labels.size(1)
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, sliding_window):
            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids, use_cache=False)
                neg_log_likelihood = outputs.loss

            # Sparsity calculation
            if model.config.seerattn_sparsity_method == 'threshold':
                sparsity.append(0.0) ## use PROFILE_FILE env variable to get the sparsity
            else:  # nz_ratio
                sparsity.append(model.config.seerattn_nz_ratio)

            outputs = None
            input_ids = None
            target_ids = None
            gc.collect()
            torch.cuda.empty_cache() 

            nlls.append(neg_log_likelihood)
            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    avg_sparsity = sum(sparsity)/len(sparsity) if sparsity else 0
    return {"mean_perplexity": ppl, "sparsity": avg_sparsity}


def main(args):
    model_id = args.model
    model_seer_path = MODEL_ID2GATE[model_id]
    model_path = MODEL_ID2PATH[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    _, testenc = get_dataset(
        data_id=args.dataset,
        train_nsamples=128,
        seed=args.seed,
        seqlen=4096,
        tokenizer=tokenizer,
        model_id=model_id,
        test_only=True
    )

    results = []
    # Model loading with config parameters
    if args.use_seer:
        if "llama" in model_seer_path.lower():
            model = SeerAttnLlamaForCausalLM.from_pretrained(
                model_seer_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                seerattn_sparsity_method=args.sparsity_method,
                seerattn_threshold=float(args.threshold.split(",")[0]),
                seerattn_nz_ratio=float(args.nz_ratios.split(",")[0]),
                seerattn_gate_type=args.gate_type,
                seerattn_last_block_dense=False,
            )
        elif "qwen" in model_seer_path.lower():
            model = SeerAttnQwen2ForCausalLM.from_pretrained(
                model_seer_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                seerattn_sparsity_method=args.sparsity_method,
                seerattn_threshold=float(args.threshold.split(",")[0]),
                seerattn_nz_ratio=float(args.nz_ratios.split(",")[0]),
                seerattn_gate_type=args.gate_type,
                seerattn_last_block_dense=False,
            )
        else:
            raise ValueError("Model unsupported for SeerAttn")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            attn_implementation="flash_attention_2",
        )

    import sys
    sys.path.append("..")
    from quant.quantizer import AttnQuantizer
    attn_quantizer = AttnQuantizer.from_qstr(args.attn_qstr)
    AttnQuantizer.plug_into_model(attn_quantizer, model.model)
    qtag = f"_{args.attn_qstr}"
    if not attn_quantizer.is_quant(): qtag = ""
    
    for seqlen in args.length:
        save_file = f"{args.save_dir}/{model_id}-{args.dataset}-{seqlen}.jsonl"
        if args.use_seer:
            params = (args.threshold.split(",") if args.sparsity_method == 'threshold' 
                        else args.nz_ratios.split(","))
            for param_val in params:
                if args.sparsity_method == 'threshold':
                    model.config.seerattn_threshold = float(param_val)
                else:
                    model.config.seerattn_nz_ratio = float(param_val)
                
                ppl = eval_ppl(testenc, model, seqlen)
                seer_tag = f"seer-{args.sparsity_method}-{float(param_val):.2f}"
                seer_tag += qtag
                append_with_lock(save_file, json.dumps({f"{seer_tag}": f"{ppl:.3f}", "seqlen": seqlen}))
        else:
            ppl = eval_ppl(testenc, model, seqlen)
            append_with_lock(save_file, json.dumps({f"full": f"{ppl:.3f}", "seqlen": seqlen}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Original arguments
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--use_seer", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length", nargs="+", type=int, default=[4096])

    # New sparsity parameters
    parser.add_argument("--sparsity_method", choices=['threshold', 'nz_ratio'], default='threshold')
    parser.add_argument("--threshold", type=str, default="0.001")
    parser.add_argument("--nz_ratios", type=str, default="0.5")
    parser.add_argument("--gate_type", type=str, default="Qavg_Kmaxminavg")

    # New quant parameters
    # e.g. q4_k4_v4_g-1_sym_rtn; qf8_kf8_vf8_g-1_sym_had;
    parser.add_argument("--attn_qstr", type=str, default="q16_k16_v16_g-1_sym_rtn")
    
    args = parser.parse_args(
        # [
        #     "--model", "qwen2.5-3b",
        #     "--dataset", "wiki",
        #     "--save_dir", "../../eval-out/ppl",
        #     "--length", "4096",
        #     "--use_seer",
        #     "--gate_type", "Qavg_Kmaxminavg",
        #     "--sparsity_method", "nz_ratio",
        #     "--nz_ratios", "0.5",
        # ]
    )
    transformers.set_seed(args.seed)
    
    # import debugpy
    # debugpy.listen(45678)
    # debugpy.wait_for_client()
    
    main(args)