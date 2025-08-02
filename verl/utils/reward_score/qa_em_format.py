# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random

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
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # is_valid_format, _ = is_valid_sequence(solution_str)
    # retrieval_correct = False
    # if is_valid_format:
    #     retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if structure_format_score == 0:
        final_reward = 1 if cover_exact_match(answer, ground_truth['target']) else 0
    else:
        final_reward = search_r1_format(solution_str, ground_truth['target'], lambda_f=structure_format_score)
    
    if do_print:
        print(f"--------------------------------")
        if structure_format_score == 0:
            print("Test Sample")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Final reward: {final_reward}")
        print(f"Solution string: {solution_str}")
            
    # if answer is None:
    #     if is_valid_format:
    #         if retrieval_correct:
    #             return structure_format_score + retrieval_score # 0.3
    #         else:
    #             return structure_format_score # 0.2
    #     else:
    #         return 0
    # else:
    #     if em_check(answer, ground_truth['target']):
    #         if is_valid_format:
    #             return score # 1
    #         else:
    #             return score - structure_format_score # 0.8
    #     elif is_valid_format:
    #         if retrieval_correct:
    #             return structure_format_score + retrieval_score # 0.3
    #         else:
    #             return structure_format_score # 0.2
    #     else:
    #         return final_format_score # 0.1
    return final_reward


def normalize_string(str):
    """Normalize answer by removing articles, punctuation, extra spaces, and lowercasing."""
    # Remove articles, punctuation, lowercase, and fix whitespace in one pass
    str = re.sub(r'\b(a|an|the)\b', ' ', str.lower())
    str = ''.join(c if c not in string.punctuation else ' ' for c in str)
    return ' '.join(str.split())


def cover_exact_match(pred, gold):
    """Check if any golden answer is a substring of the prediction."""
    if not pred:
        return False
    if isinstance(gold, str):
        gold = [gold]
    norm_pred = normalize_string(pred)
    return any(normalize_string(ans) in norm_pred for ans in gold)


def extract_str_between(text, start, end):
    """Extract strings between start and end markers."""
    results = []
    start_idx = 0
    while True:
        start_idx = text.find(start, start_idx)
        if start_idx == -1:
            break
        start_idx += len(start)
        end_idx = text.find(end, start_idx)
        if end_idx == -1:
            break
        results.append(text[start_idx:end_idx])
        start_idx = end_idx + len(end)
    return results


def check_full_response_format(full_response: str) -> bool:
    """
    Check if the model response is in the correct format as specified in AGENT_PROMPT_V2 for agentic rag LLM.
    Returns True if the format is correct, otherwise False.
    Accepts two step formats:
      1. <step><reasoning>...</reasoning><search>...</search><context>...</context><conclusion>...</conclusion></step>
      2. <step><reasoning>...</reasoning><conclusion>...</conclusion></step>
    """
    # Find the last occurrence of <think>
    idx = full_response.rfind('<think>')
    if idx == -1:
        return False
    response = full_response[idx:].strip()
    # Regex for the overall structure: <think>...</think><answer>...</answer>
    pattern = re.compile(
        r'^<think>\s*((<step>.*?</step>\s*)+)</think>\s*<answer>.*?</answer>\s*$',
        re.DOTALL
    )
    match = pattern.match(response)
    if not match:
        return False
    # Extract all <step> blocks
    steps = re.findall(r'<step>(.*?)</step>', match.group(1), re.DOTALL)
    if not steps:
        return False
    # Check each <step> for required tags in order, no duplicates, and valid format
    for step in steps:
        # Find all tags in order
        tags = re.findall(r'<(reasoning|search|context|conclusion)>', step)
        # Acceptable tag sequences:
        # 1. ['reasoning', 'search', 'context', 'conclusion']
        # 2. ['reasoning', 'conclusion']
        if tags == ['reasoning', 'search', 'context', 'conclusion']:
            # Ensure each tag appears only once
            for tag in ['reasoning', 'search', 'context', 'conclusion']:
                if step.count(f'<{tag}>') != 1 or step.count(f'</{tag}>') != 1:
                    return False
            # Ensure <search> and <context> are non-empty
            for tag in ['search', 'context']:
                content = re.search(f'<{tag}>(.*?)</{tag}>', step, re.DOTALL)
                if content is None or content.group(1).strip() == '':
                    return False
            # Ensure no extra tags between <step> and </step> by removing the four required tags and their content
            cleaned = step
            for tag in ['reasoning', 'search', 'context', 'conclusion']:
                cleaned = re.sub(f'<{tag}>.*?</{tag}>', '', cleaned, flags=re.DOTALL)
            if re.search(r'</?[^>]+>', cleaned):
                return False
        elif tags == ['reasoning', 'conclusion']:
            # Ensure each tag appears only once
            for tag in ['reasoning', 'conclusion']:
                if step.count(f'<{tag}>') != 1 or step.count(f'</{tag}>') != 1:
                    return False
            # Ensure no extra tags between <step> and </step> by removing the two required tags and their content
            cleaned = step
            for tag in ['reasoning', 'conclusion']:
                cleaned = re.sub(f'<{tag}>.*?</{tag}>', '', cleaned, flags=re.DOTALL)
            if re.search(r'</?[^>]+>', cleaned):
                return False
        else:
            return False
    return True


def check_full_response_format_optimized(full_response: str) -> bool:
    """
    Optimized version of check_full_response_format.
    Check if the model response is in the correct format as specified in AGENT_PROMPT_V2 for agentic rag LLM.
    Returns True if the format is correct, otherwise False.
    Accepts two step formats:
      1. <step><reasoning>...</reasoning><search>...</search><context>...</context><conclusion>...</conclusion></step>
      2. <step><reasoning>...</reasoning><conclusion>...</conclusion></step>
    """
    # Find the last occurrence of <think>
    idx = full_response.rfind('<think>')
    if idx == -1:
        return False
    
    # Extract the relevant part and strip whitespace
    response = full_response[idx:].strip()
    
    # Quick checks for required structure using string methods instead of regex
    if not response.startswith('<think>'):
        return False
    
    # Find the end of think tag
    think_end = response.find('</think>')
    if think_end == -1:
        return False
    
    # Check for answer tag after think - must handle whitespace
    answer_section = response[think_end + 8:].lstrip()  # Skip '</think>' and whitespace
    if not answer_section.startswith('<answer>'):
        return False
    
    answer_end = answer_section.find('</answer>')
    if answer_end == -1:
        return False
    
    # Check no extra content after </answer>
    remaining = answer_section[answer_end + 9:].strip()
    if remaining:
        return False
    
    # Extract think content (handle potential whitespace)
    think_start = 7  # Length of '<think>'
    # Skip whitespace after <think>
    while think_start < think_end and response[think_start].isspace():
        think_start += 1
    
    think_content = response[think_start:think_end].rstrip()
    
    # Check if there's at least one step
    if '<step>' not in think_content or '</step>' not in think_content:
        return False
    
    # Process each step
    step_start = 0
    while True:
        # Find next step
        step_tag_start = think_content.find('<step>', step_start)
        if step_tag_start == -1:
            break
        
        step_tag_end = think_content.find('</step>', step_tag_start)
        if step_tag_end == -1:
            return False
        
        # Extract step content
        step_content = think_content[step_tag_start + 6:step_tag_end]
        
        # Check step format using a more efficient approach
        if not validate_step_optimized(step_content):
            return False
        
        step_start = step_tag_end + 7  # Move past '</step>'
    
    return True


def validate_step_optimized(step_content: str) -> bool:
    """
    Optimized validation of a single step's content.
    Returns True if the step follows one of the two valid formats.
    """
    # Quick check for reasoning tag (required in both formats)
    reasoning_start = step_content.find('<reasoning>')
    if reasoning_start == -1:
        return False
    
    reasoning_end = step_content.find('</reasoning>', reasoning_start)
    if reasoning_end == -1:
        return False
    
    # Check what comes after reasoning
    after_reasoning = step_content[reasoning_end + 12:]
    
    # Check for search tag (format 1)
    if '<search>' in after_reasoning:
        # Format 1: reasoning, search, context, conclusion
        search_start = after_reasoning.find('<search>')
        search_end = after_reasoning.find('</search>', search_start)
        if search_end == -1:
            return False
        
        # Check if search is non-empty
        search_content = after_reasoning[search_start + 8:search_end].strip()
        if not search_content:
            return False
        
        # Check for context
        after_search = after_reasoning[search_end + 9:]
        context_start = after_search.find('<context>')
        if context_start == -1:
            return False
        
        context_end = after_search.find('</context>', context_start)
        if context_end == -1:
            return False
        
        # Check if context is non-empty
        context_content = after_search[context_start + 9:context_end].strip()
        if not context_content:
            return False
        
        # Check for conclusion
        after_context = after_search[context_end + 10:]
        conclusion_start = after_context.find('<conclusion>')
        if conclusion_start == -1:
            return False
        
        conclusion_end = after_context.find('</conclusion>', conclusion_start)
        if conclusion_end == -1:
            return False
        
        # Verify tag counts using a more efficient method
        tag_counts = count_tags_optimized(step_content)
        if (tag_counts.get('reasoning', 0) != 1 or
            tag_counts.get('search', 0) != 1 or
            tag_counts.get('context', 0) != 1 or
            tag_counts.get('conclusion', 0) != 1):
            return False
        
        # Check for extra tags
        if has_extra_tags_format1(step_content):
            return False
        
    else:
        # Format 2: reasoning, conclusion
        conclusion_start = after_reasoning.find('<conclusion>')
        if conclusion_start == -1:
            return False
        
        conclusion_end = after_reasoning.find('</conclusion>', conclusion_start)
        if conclusion_end == -1:
            return False
        
        # Verify tag counts
        tag_counts = count_tags_optimized(step_content)
        if (tag_counts.get('reasoning', 0) != 1 or
            tag_counts.get('conclusion', 0) != 1):
            return False
        
        # Check for extra tags
        if has_extra_tags_format2(step_content):
            return False
    
    return True


def count_tags_optimized(content: str) -> dict:
    """
    Efficiently count occurrences of specific tags.
    """
    tags = ['reasoning', 'search', 'context', 'conclusion']
    counts = {}
    
    for tag in tags:
        open_tag = f'<{tag}>'
        close_tag = f'</{tag}>'
        
        # Count occurrences
        open_count = content.count(open_tag)
        close_count = content.count(close_tag)
        
        if open_count > 0 or close_count > 0:
            counts[tag] = open_count
    
    return counts


def has_extra_tags_format1(content: str) -> bool:
    """
    Check for extra tags in format 1 (with all 4 tags).
    Optimized version using string operations instead of regex.
    """
    # Create a mutable copy of content
    cleaned = content
    
    # Remove all valid tag pairs and their content
    for tag in ['reasoning', 'search', 'context', 'conclusion']:
        start_tag = f'<{tag}>'
        end_tag = f'</{tag}>'
        
        while True:
            start_idx = cleaned.find(start_tag)
            if start_idx == -1:
                break
            
            end_idx = cleaned.find(end_tag, start_idx)
            if end_idx == -1:
                break
            
            # Remove the tag and its content
            cleaned = cleaned[:start_idx] + cleaned[end_idx + len(end_tag):]
    
    # Check if any tags remain by looking for < and > characters
    return '<' in cleaned and '>' in cleaned


def has_extra_tags_format2(content: str) -> bool:
    """
    Check for extra tags in format 2 (only reasoning and conclusion).
    Optimized version using string operations instead of regex.
    """
    # Create a mutable copy of content
    cleaned = content
    
    # Remove all valid tag pairs and their content
    for tag in ['reasoning', 'conclusion']:
        start_tag = f'<{tag}>'
        end_tag = f'</{tag}>'
        
        while True:
            start_idx = cleaned.find(start_tag)
            if start_idx == -1:
                break
            
            end_idx = cleaned.find(end_tag, start_idx)
            if end_idx == -1:
                break
            
            # Remove the tag and its content
            cleaned = cleaned[:start_idx] + cleaned[end_idx + len(end_tag):]
    
    # Check if any tags remain by looking for < and > characters
    return '<' in cleaned and '>' in cleaned


def search_r1_format(model_response, ground_truth_answer, lambda_f=0.4):
    """
    Implements the reward function of Search R1 from Equation 3 in Section 4.1 in the paper "An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents" (https://arxiv.org/abs/2505.15117).
    
    Args:
        model_response: The complete model response string
        ground_truth_answer: The correct answer (string or list of strings)
        lambda_f: Format reward weight (default 0.4 for 7B models)
    
    Returns:
        float: Reward score between 0 and 1
    """
    # Extract predicted answer
    answer_matches = extract_str_between(model_response, '<answer>', '</answer>')
    predicted_answer = answer_matches[-1].strip() if answer_matches else ""
    
    # Check correctness and format
    answer_correct = cover_exact_match(predicted_answer, ground_truth_answer)
    format_correct = check_full_response_format_optimized(model_response)
    
    # Apply Equation 3
    if answer_correct:
        return 1.0 if format_correct else 1.0 - lambda_f
    else:
        return lambda_f if format_correct else 0.0
