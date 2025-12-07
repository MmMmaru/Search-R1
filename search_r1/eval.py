import re
import string
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
import torchdata 

import requests

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score
def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False
    
def extract(text):
    "return answer, search contents"
    pattern1 = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    pattern2 = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    answer_matches = pattern1.findall(text)
    search_matches = pattern2.findall(text)
    answer = answer_matches[-1].strip() if answer_matches else None
    search = search_matches[-1].strip() if search_matches else None
    return answer, search

curr_eos = [151645, 151643] # for Qwen2.5 series models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    model_path = '/root/autodl-tmp/models/Qwen/Qwen3-0.6B'
    dataset_path = "data/nq_search/test.parquet"
    model = AutoModelForCausalLM.from_pretrained(
        model_path
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    test_dataset = Dataset.from_parquet(dataset_path)

    # Initialize the stopping criteria
    search_tokens = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    answer_tokens = ["</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n"]
    information_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(search_tokens+answer_tokens, tokenizer)])

    score = 0
    max_truns = 3
    answer_content = None
    for i in range(200):
        print("================= Example", i, "=================")
        item = test_dataset[i]
        prompt = item['prompt']
        print("\nQuestion:", prompt[0]['content'])
        prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        input_text = prompt
        for _ in range(max_truns):
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            attention_mask = torch.ones((1, input_ids.shape[1]), dtype=torch.long).to(device)
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                # stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            output_ids = output_ids[:, input_ids.shape[1]:]
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print("\nModel output:", output_text)

            answer_content, search_content = extract(output_text)
            if answer_content:
                break
            if search_content:
                retrieved_contents = search(search_content)
                print("\nRetrieved contents:", retrieved_contents)
                input_text += information_template.format(output_text=output_text, search_results=retrieved_contents)
            else:
                input_text += output_text
        if answer_content:
            print("\nFinal Answer:", answer_content)
            print("\nGround Truth:", item['reward_model']['ground_truth']['target'])
            em_score = em_check(answer_content, item['reward_model']['ground_truth']['target'])
            score += em_score
            print("\nEM Score for this example:", em_score)
    print("\nFinal EM Score:", score / 200)
    import os
    with open("eval_trained.txt", "a") as f:
        f.write(f"\n{model_path}: - EM Score: {score / 200}\n")

if __name__ == "__main__":
    main()