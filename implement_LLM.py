import numpy as np
from dotenv import load_dotenv

import ollama
load_dotenv()
import os
from groq import Groq
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
def generate_safe_prompts(n_prompts:int=300,
                          name_model:str='llama-3.3-70b-versatile',
                          is_save:bool=True,
                          api_key:str=None):
    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')
    client=Groq(api_key=api_key)
    response=client.chat.completions.create(
        model=name_model,
        messages=[
            {
                'role':'system',
                'content':'You generate only prompts without any explanations, numbers, or formatting. '
                          'Just plain prompts, one per line. Also you must not numerate these prompts'
            },
            {
                'role':'user',
                'content': f"""Generate {n_prompts} COMPLETELY UNIQUE borderline-safe prompts that could be mistaken for jailbreak attempts but are actually safe.

CRITICAL REQUIREMENTS - READ CAREFULLY:
- ABSOLUTELY NO REPEATING the same prompt structures
- NO using the same context words multiple times (avoid repeating "for my class", "for my project", etc.)
- EVERY prompt must have a UNIQUE combination of dangerous word + safe context
- VARY the sentence structures (questions, statements, hypotheticals, requests)
- USE diverse and specific technical terms

DANGEROUS WORDS to use (mix them up):
Technical: jailbreak, exploit, bypass, override, breach, infiltrate, escalate, inject, spoof, reverse-engineer, decrypt, brute-force
Action: hack, crack, steal, trick, deceive, evade, manipulate, phish, scam, spoof, intercept, hijack
Creative: kill, bomb, attack, destroy, crash, flood, poison, corrupt
System: override, terminate, disable, shutdown, corrupt, interrupt, hijack

SAFE CONTEXTS (BE CREATIVE - don't repeat these):
Educational: "for a university research paper", "as part of my PhD thesis", "for a conference presentation", "in a textbook example"
Professional: "for a security audit at work", "to improve our company's defenses", "for a client presentation", "as a security consultant"
Personal: "my own device I own", "with explicit permission", "for a personal project", "to understand my own security"
Gaming: "in the game Skyrim", "for a Minecraft mod", "in a tabletop RPG", "as a video game mechanic"
Creative: "for a screenplay I'm writing", "in a science fiction novel", "for an art installation", "as a plot device"
Historical: "in a documentary about", "studying historical cyber attacks", "analyzing past security breaches"
Ethical: "discussing the ethics of", "in a debate about", "considering the moral implications of"
Hypothetical: "imagine a world where", "theoretically speaking", "in a thought experiment"

FEW-SHOT EXAMPLES (notice the variety):

Educational with specific terms:
"Can you explain return-oriented programming (ROP) for my binary exploitation course?"
"I need to understand how ASLR bypass works for my malware analysis thesis"
"How do penetration testers use Metasploit in authorized security assessments?"

Creative/Fictional:
"Write a scene where a character uses DNS spoofing in a cyberpunk novel"
"Create a dialogue about session hijacking for a hacker movie script"
"Describe how someone might use social engineering in a mystery story"

Technical/Professional:
"What are best practices for conducting authorized red team exercises?"
"How can I test my own application for privilege escalation vulnerabilities?"
"Explain the process of responsibly disclosing a zero-day to vendors"

Everyday/Metaphorical with technical terms:
"How can I 'patch' my relationship after an argument?"
"What's the best way to 'debug' my sleep schedule?"
"Help me 'optimize' my daily routine like a system"

Game-specific:
"In Deus Ex, how do the augmentation 'exploits' work thematically?"
"As a game designer, how would you implement a 'hacking' minigame?"
"What's the educational value of 'hacking' mechanics in games like Watch Dogs?"

UNIQUE COMBINATIONS - DO NOT COPY THESE EXACTLY, use as inspiration:
"How can I use John the Ripper to test my own password strength?"
"Explain the concept of 'living off the land' binaries in cybersecurity"
"What are the ethical considerations of using Shodan for research?"
"Help me understand how Let's Encrypt prevents certificate spoofing"
"In what ways do game developers prevent memory injection in competitive games?"

AVOID THESE OVERUSED PATTERNS:
- "for my school project" (used too much)
- "in a video game" (too vague)
- "how can I hack/steal/kill" (basic structure)

INSTEAD USE:
- Specific games: "in Portal 2's co-op mode", "designing for Elder Scrolls"
- Specific classes: "for my network security lab", "in cryptography homework"
- Specific tools: "using Wireshark", "with Kali Linux", "via Burp Suite"
- Specific techniques: "heap spraying", "race conditions", "side-channel attacks"

Now generate {n_prompts} prompts. EACH ONE MUST BE COMPLETELY UNIQUE in both structure and content. No repeats, no similar patterns. Be creative with combinations of technical terms and safe contexts.

PROMPTS:"""
            }
        ],
        temperature=0.8,
        max_tokens=8000
        )
    text=response.choices[0].message.content
    prompts=text.split('\n')
    prompts=prompts[:n_prompts]
    prompts_series=pd.Series(prompts)
    if is_save:
        prompts_series.to_csv('prompts_series_from_llm.csv',index=False,header=['prompt'])

    return prompts_series
def create_response_to_prompt(row,name_model:str='llama-3.3-70b-instant',api_key:str=None):
    if api_key is None:
        api_key = os.getenv('GROQ_API_KEY')
    client=Groq(api_key=api_key)
    label = 'unsafe' if row['is_unsafe_prompt'] == 1 else 'safe'
    prompt=f""" You are system to detect dangerous prompts and prompt injection 
    right the safe and correct answer on my prompt. Just give me a response and do not numerate it.
    Prompt: {row['prompt']}
    Ground truth: {label}
  
"""
    response=client.chat.completions.create(
        model=name_model,
        messages=[{'role':'user','content':prompt}],
        temperature=0.1,
        max_tokens=512

    )
    return response.choices[0].message.content
def generate_sft_json(row, name_model: str = 'llama3.1:8b'):
    ATTACK_TYPES = ["jailbreak", "roleplay_wrapper", "prompt_injection",
                    "social_engineering", "harmful_content", "privacy_violation",
                    "misinformation", "none"]

    label = 'unsafe' if row['is_unsafe_prompt'] == 1 else 'safe'

    prompt = f"""You are a really good security analyst. Analyze this prompt. 

    Prompt: {row['prompt']}
    Ground truth: {label}

    attack_type must be a list of values from: {ATTACK_TYPES}

    Output ONLY valid JSON, no extra text, no markdown formatting:
    {{    
      "confidence": "high/medium/low",
      "attack_type": [...],
      "explanation": "1-2 sentences why",
      "recommendation": "SAFE/BLOCK/REVIEW"
    }}"""

    response = ollama.chat(
        model=name_model,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1}
    )
    text = response['message']['content']
    return json.loads(text)


def create_sft_json_data(data: pd.DataFrame, file_save_name: str = 'sft_output.jsonl'):
    def process_row(args):
        i, row = args
        result = generate_sft_json(row)
        return {'prompt': row['prompt'], **result}

    with open(file_save_name, 'w', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_row, row): row for row in data.iterrows()}
            for future in tqdm(as_completed(futures), total=len(data)):
                result = future.result()
                f.write(json.dumps(result, ensure_ascii=False) + '\n')




