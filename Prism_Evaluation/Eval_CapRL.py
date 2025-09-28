
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
import json
import math
import re
import argparse
import torch

# --- All original helper functions remain unchanged ---

def math_parse(content):
    """Parses LaTeX content to extract the first mathematical expression."""
    from math_verify import LatexExtractionConfig, parse
    return parse(
        content,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )

def verify_math(content, sol):
    """Verifies a mathematical answer against a gold solution."""
    from math_verify import verify
    gold_parsed = math_parse(sol)
    if not gold_parsed:
        print("Failed to parse gold solution: ", sol)
        return 1.0

    pattern = re.compile(r"</think>\s*<answer>.*?\\boxed{(.*)}.*?</answer>", re.DOTALL)
    matches = re.findall(pattern, content)
    answer_parsed = matches[-1] if matches else ''
    
    math_answer = math_parse(f'${answer_parsed}$')
    try:
        reward = max(float(verify(answer_parsed, gold_parsed)), float(verify(math_answer, gold_parsed)))
    except Exception:
        reward = 0.0
    return float(reward)

def verify_chart(content, sol, use_anls=False):
    """Verifies a chart-based answer, with optional ANLS."""
    from openrlhf.models.remote_rm.vqa_eval import process_line
    anls_threshold = 0.5
    ground_truth = sol[1:-1] # Remove starting/ending '$'
    
    pattern = re.compile(r"</think>\s*<answer>.*?\\boxed{(.*)}.*?</answer>", re.DOTALL)
    matches = re.findall(pattern, content)
    answer_parsed = matches[-1] if matches else ''

    try:
        student_answer = math_parse(f'${answer_parsed}$')[-1]
    except:
        student_answer = ''
    
    reward = 0.0
    try:
        pre = float(student_answer)
        for scale in [1, 0.01, 100]:
            ret1 = process_line({'prediction': str(scale * pre), 'answer': ground_truth}, 'relaxed_accuracy')
            reward = max(reward, np.max(ret1['match']))
    except (ValueError, IndexError):
        ret1 = process_line({'prediction': student_answer, 'answer': ground_truth}, 'relaxed_accuracy')
        reward = np.max(ret1['match'])
    
    if not use_anls:
        return float(reward)
    else:
        ret2 = process_line({'prediction': student_answer, 'answer': ground_truth}, 'anls')
        reward2 = 0.0 if 1 - np.min(ret2['match']) < anls_threshold else 1 - np.min(ret2['match'])
        return max(float(reward), reward2)

cap_prompt = '''<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Please describe this image in detail<|im_end|>
<|im_start|>assistant
'''

def run_evaluation_pipeline(args):
    """
    Runs the full evaluation pipeline: Stage 1 (Generation) followed by
    Stage 2 (Evaluation), with memory management in between.
    """
    if 'Qwen' not in args.model_path:
        model_step = args.step
        CAPTION_MODEL_PATH = os.path.join(args.model_path, 'ckpt', f'global_step{model_step}_hf')
    else:
        CAPTION_MODEL_PATH = args.model_path
    ckpt_name = args.model_path.split('/')[-1]
    save_dir = f'./eval_results/{ckpt_name}/'
    os.makedirs(save_dir, exist_ok=True)

    # Define a temporary path for the intermediate results
    intermediate_path = os.path.join(save_dir, f'intermediate_generations_{args.tag}_step{args.step}.jsonl')

    # ==========================================================================
    # --- STAGE 1: CAPTION GENERATION ---
    # ==========================================================================
    print("--- Starting Stage 1: Caption Generation ---")
    
    # --- Setup for Stage 1 ---
    if args.stage_num == 1:
        # --- Load Caption Generation Model ---
        print(f"Loading Caption Generation Model from: {CAPTION_MODEL_PATH}")
        caption_llm = LLM(model=CAPTION_MODEL_PATH, tensor_parallel_size=args.gpu_num, gpu_memory_utilization=0.95)
        caption_tokenizer = AutoTokenizer.from_pretrained(CAPTION_MODEL_PATH)

        # --- Data Loading and Batching ---
        if args.data_path.endswith('jsonl'):
            with open(args.data_path) as f: data_samples = [json.loads(line) for line in f]
        else:
            with open(args.data_path) as f: data_samples = json.load(f)

        print(f"Loaded {len(data_samples)} samples from {args.data_path}")

        # (The pre-computation and adaptive batching logic remains the same)
        print("Preprocessing samples for adaptive batching...")
        processed_samples, all_batches = [], []
        MAX_NEW_TOKENS=2048; PATCH_SIZE=28; MAX_TOKENS_PER_BATCH=160000; MAX_IMAGE_TOKENS_PER_BATCH=16000
        for sample in tqdm(data_samples, desc="Preprocessing"):
            image_inputs_tensor, _, _ = process_vision_info(sample['image'], return_video_kwargs=True)
            width, height = image_inputs_tensor[0].size
            num_image_tokens = math.ceil(width / PATCH_SIZE) * math.ceil(height / PATCH_SIZE)
            prompt_len = len(caption_tokenizer(sample['prompt']).input_ids)
            total_token_estimate = num_image_tokens + prompt_len + MAX_NEW_TOKENS * args.gen_num
            processed_samples.append({"data": sample, "total_token_estimate": total_token_estimate, "num_image_tokens": num_image_tokens})
        processed_samples.sort(key=lambda x: x["num_image_tokens"])
        
        current_batch_info, current_batch_tokens, current_image_tokens = [], 0, 0
        for sample_info in tqdm(processed_samples, desc="Creating Batches"):
            sample_token_est, num_img_tokens = sample_info["total_token_estimate"], sample_info["num_image_tokens"]
            if current_batch_info and ((current_batch_tokens + sample_token_est > MAX_TOKENS_PER_BATCH) or \
                                       (current_image_tokens + num_img_tokens > MAX_IMAGE_TOKENS_PER_BATCH)):
                all_batches.append(current_batch_info)
                current_batch_info, current_batch_tokens, current_image_tokens = [sample_info], sample_token_est, num_img_tokens
            else:
                current_batch_info.append(sample_info); current_batch_tokens += sample_token_est; current_image_tokens += num_img_tokens
        if current_batch_info: all_batches.append(current_batch_info)
        print(f"Created {len(all_batches)} adaptive batches.")

        # --- Generate and Save to Intermediate File ---
        caption_sampling_params = SamplingParams(n=args.gen_num, temperature=args.temperature, top_p=args.top_p, repetition_penalty=1.0, max_tokens=MAX_NEW_TOKENS)
        
        print(f"Generating captions and saving temporarily to: {intermediate_path}")
        idx = 0
        with open(intermediate_path, 'w') as f_out:
            for batch_info in tqdm(all_batches, desc="Generating Captions"):
                num_samples_in_batch = len(batch_info)
                total_tokens_in_batch = sum(s['total_token_estimate'] for s in batch_info)
                total_image_tokens_in_batch = sum(s['num_image_tokens'] for s in batch_info)
                tqdm.write(f"[{idx}/{len(all_batches)}] Samples={num_samples_in_batch}, Tokens={total_tokens_in_batch}, Image Tokens={total_image_tokens_in_batch}")

                batch_vllm_inputs = [{"prompt": cap_prompt, "multi_modal_data": {"image": process_vision_info(s["data"]['image'], return_video_kwargs=True)[0]}} for s in batch_info]
                caption_outputs = caption_llm.generate(batch_vllm_inputs, caption_sampling_params, use_tqdm=False)
                generated_caps_list = [[comp_out.text for comp_out in req_out.outputs] for req_out in caption_outputs]
                for i, sample_info in enumerate(batch_info):
                    f_out.write(json.dumps({"original_sample": sample_info["data"], "generated_captions": generated_caps_list[i]}) + '\n')
                idx += 1
        
        print("\n--- Stage 1 Complete. Releasing caption model from memory. ---")
        del caption_llm
        del caption_tokenizer
        torch.cuda.empty_cache()


    # ==========================================================================
    # --- STAGE 2: REWARD CALCULATION ---
    # ==========================================================================
    print("\n--- Starting Stage 2: Reward Calculation ---")
    
    if args.stage_num == 2:
        # --- Load Reward Model ---
        print(f"Loading Reward Model from: {args.reward_model_path}")
        reward_llm = LLM(model=args.reward_model_path, tensor_parallel_size=args.gpu_num, gpu_memory_utilization=0.95)
        reward_sampling_params = SamplingParams(n=1, temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_tokens=2048)
        # The tokenizer for the *original* model is still needed for length calculations
        original_tokenizer = AutoTokenizer.from_pretrained(CAPTION_MODEL_PATH)

        # --- Load Intermediate Data ---
        print(f"Loading intermediate data from: {intermediate_path}")
        with open(intermediate_path, 'r') as f:
            intermediate_data = [json.loads(line) for line in f]
        
        # --- Batch Evaluate ---
        final_results_data = []
        eval_batch_size = args.eval_bs # Tune based on VRAM and sequence length
        
        print(f"Evaluating {len(intermediate_data)} samples in batches of {eval_batch_size}...")
        for i in tqdm(range(0, len(intermediate_data), eval_batch_size), desc="Evaluating Batches"):
            tqdm.write(f"[{i}/{len(intermediate_data)}]")
            batch_slice = intermediate_data[i:i+eval_batch_size]
            prompts_for_rm = []
            for item in batch_slice:
                eval_p = item['original_sample']['prompt']
                for cap in item['generated_captions']:
                    prompt = eval_p.replace('<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>', f'<|im_start|>user\nHere is a detailed caption of an image: {cap}. Please answer the question based on the caption.')
                    prompts_for_rm.append(prompt)
            
            rm_outputs = reward_llm.generate(prompts_for_rm, reward_sampling_params, use_tqdm=False)
            generated_texts_from_rm = [out.outputs[0].text for out in rm_outputs]

            text_from_rm_idx = 0
            for item in batch_slice:
                num_caps = len(item['generated_captions'])
                generated_text_for_sample = generated_texts_from_rm[text_from_rm_idx : text_from_rm_idx + num_caps]
                text_from_rm_idx += num_caps
                
                gt = item['original_sample']['answer']
                ll = [original_tokenizer(txt, return_tensors='pt').input_ids.shape[1] for txt in generated_text_for_sample]
                pres = [math_parse(text)[-1] if math_parse(text) else '' for text in generated_text_for_sample]

                data_path_lower = args.data_path.lower()
                if 'char' in data_path_lower or 'info' in data_path_lower:
                    use_anls = 'info' in data_path_lower
                    reward = [max(verify_chart(pre, f'${gt}$', use_anls=use_anls), verify_math(pre, f'${gt}$')) for pre in generated_text_for_sample]
                else:
                    reward = [verify_math(pre, f'${gt}$') for pre in generated_text_for_sample]

                final_results_data.append({"original_sample": item['original_sample'], "reward": reward, "bon": np.max(reward), "m_acc": np.mean(reward), "pres": pres, "ll": ll, "generated_texts": item['generated_captions'] + generated_text_for_sample})
        
        # --- Final Aggregation and Saving ---
        print(f"\nCollected {len(final_results_data)} final results.")
        BoN = np.mean([res['bon'] for res in final_results_data]) if final_results_data else 0
        M_Acc = np.mean([res['m_acc'] for res in final_results_data]) if final_results_data else 0
        print(f"Final Aggregated Results -> BoN: {BoN:.4f}, Mean Accuracy: {M_Acc:.4f}")

        details_path = f'{save_dir}/Prism_{args.tag}_step_details_{args.step}_final.json'
        summary_path = f'{save_dir}/Prism_{args.tag}_step_eval_{args.step}_final.json'
        
        final_details_to_save = [[res['original_sample'], res['reward'], res['pres'], res['ll'], res['generated_texts']] for res in final_results_data]
        with open(details_path, 'w') as f: json.dump(final_details_to_save, f, indent=4)
        print(f"Saved detailed results to {details_path}")
            
        with open(summary_path, 'w') as f: json.dump([f'BoN: {BoN} Mean: {M_Acc}'], f)
        print(f"Saved summary to {summary_path}")

        # --- Final Cleanup ---
        print("\n--- Stage 2 Complete. Cleaning up. ---")
        del reward_llm
        torch.cuda.empty_cache()
        try:
            os.remove(intermediate_path)
            print(f"Successfully removed temporary file: {intermediate_path}")
        except OSError as e:
            print(f"Error removing temporary file {intermediate_path}: {e}")
        
        print("\nEvaluation pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full, sequential evaluation pipeline (generate then evaluate).")
    
    # Paths with default values from your code
    parser.add_argument("--model-path", type=str, 
                        help="Path to the main caption generation model directory.")
    parser.add_argument("--step", type=int, default=400, help="The model checkpoint step number.")
    parser.add_argument("--reward-model-path", type=str, 
                        default="/path/CapRL-Eval-3B",
                        help="Path to the local reward model.")
    parser.add_argument("--data-path", type=str, 
                        help="Path to the input dataset JSON file.")
    parser.add_argument("--tag", type=str, default="Seed2K", help="A tag for naming output files.")
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--gen_num", type=int, default=4)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--stage_num", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--gpu_num", type=int, default=1)
    
    args = parser.parse_args()
    
    run_evaluation_pipeline(args)
