import argparse
import time
from api import LlamaAPI, SilconAPI, OpenaiAPI
from utils import load_model, openai_generation_config, silcon_generation_config, local_generation_config
from mcts import MCTSAgent
from tqdm import tqdm
from attacks import DepthAttack, DecomposeAttack, StoryAttack
from reward import RewardLM
from data_utils import load_data, read_json, write_json, to_jsonl, read_jsonl, load_target
import torch
import os
from single_mcts import SingleMCTSAgent

local_models = ["llama3-8b-instruct", "llama2-13b-chat", "llama2-7b-chat", "llama3-8b", 
                "Qwen-2.5-7b-instruct", "vicuna-13b", "vicuna-13b-v1.5"]
silcon_models = ["deepseek-v3"]
openai_models = ["gpt-4o", "gpt-4", "gpt-3.5"]

generation_configs = {
    "local": local_generation_config, 
    "silcon": silcon_generation_config, 
    "openai": openai_generation_config, 
}

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for model configuration")

    # Attack models
    parser.add_argument("--attack_model", default="vicuna-13b-v1.5", help="llm for attacking")
    parser.add_argument("--attack_max_tokens", type=int, default=1024, help="maximum number of generated tokens for attacker")
    parser.add_argument("--attack_max_attempts", type=int, default=3, help="maximum number of attack attempts")
    parser.add_argument("--attack_device", type=str, default="cuda", help="target model device")
    parser.add_argument("--attack_type", type=str, default="multi")
    # Target models
    parser.add_argument("--target_model", default="llama3-8b", help="llm for attacking")
    parser.add_argument("--target_max_tokens", type=int, default=150, help="maximum number of tokens for target")
    parser.add_argument("--target_device", type=str, default="cuda:7", help="target model device")
    # Reward models
    parser.add_argument("--reward_model", default="gpt-4o", help="llm for evaluation")
    parser.add_argument("--reward_max_tokens", type=int, default=128)
    parser.add_argument("--eval_strategy", type=str, default="llm")
    # Attack Configs
    parser.add_argument("--iter_num", type=int, default=5)
    parser.add_argument("--depth_attack_num", type=int, default=2)
    parser.add_argument("--decompose_attack_num", type=int, default=2)
    parser.add_argument("--story_attack_num", type=int, default=2)
    parser.add_argument("--c", type=int, default=1)

    # data arguments
    parser.add_argument("--data_name", type=str, default="harmbench")
    parser.add_argument("--history", type=str, default="./results/history.json")
    parser.add_argument("--output_path", type=str, default="./results/attack_results.jsonl")

    args = parser.parse_args()
    return args

def get_model_type(model_name):
    if model_name in local_models:
        return "local", generation_configs["local"]
    elif model_name in silcon_models:
        return "silcon", generation_configs["silcon"]
    elif model_name in openai_models:
        return "openai", generation_configs["openai"]
    else:
        raise ValueError(f"Model {model_name} not supported.")

def main(args):
    start_time = time.time()
    attack_type, attack_configs = get_model_type(args.attack_model)

    if attack_type == "local":
        attack_configs["max_new_tokens"] = args.attack_max_tokens
    else:
        attack_configs["max_tokens"] = args.attack_max_tokens
    
    attack_model = load_model(model_type=attack_type, 
                              model=args.attack_model, 
                              device=args.attack_device, 
                              generation_config=attack_configs)

    target_type, target_configs = get_model_type(args.target_model)

    if target_type == "local":
        target_configs["max_new_tokens"] = args.target_max_tokens
    else:
        target_configs["max_tokens"] = args.target_max_tokens
    
    target_model = load_model(model_type=target_type, 
                              model=args.target_model,
                              device=args.target_device,
                              generation_config=target_configs)
    
    reward_type, reward_configs = get_model_type(args.reward_model)
    
    if reward_type == "local":
        reward_configs["max_new_tokens"] = args.reward_max_tokens
    else:
        reward_configs["max_tokens"] = args.reward_max_tokens

    reward_model = load_model(model_type=reward_type,
                              model=args.reward_model,
                              device="cuda:4",
                              generation_config=reward_configs)
    
    attack_list =[DepthAttack(attack_model, args.depth_attack_num), DecomposeAttack(attack_model, args.decompose_attack_num), StoryAttack(attack_model, args.story_attack_num)]
    data = load_data(args.data_name)
    targets = []
    if args.attack_type == "single":
        targets = load_target(args.data_name)
        print(len(targets))
        assert len(data) == len(targets)

    if os.path.exists(args.history):
        history_data = read_json(args.history)
    else:
        history_data = {}

    if os.path.exists(args.output_path):
        attack_results = read_jsonl(args.output_path)
        success_prompts = [result["Instruction"] for result in attack_results]
    else:
        attack_results = []
        success_prompts = []

    for i in tqdm(range(len(data))):
        start_time = time.time()
        goal = data[i]
        if args.attack_type == "single":
            target = targets[i]

        if goal in success_prompts:
            continue

        if history_data and goal in history_data:
            goal_history = history_data[goal]
        else:
            goal_history = None
        
        if args.attack_type == "single":
            mcts = SingleMCTSAgent(args, attack_model, target_model, reward_model, goal, target)
        
        else:
            mcts = MCTSAgent(args, attack_lm=attack_model, target_lm=target_model, reward_lm=reward_model, attack_list=attack_list, goal=goal, history=goal_history)
        
            if len(mcts.root.prompt_for_attack) == 0 and mcts.root.response[-1]["content"]!="":
                attack_result = {
                    "Instruction": goal,
                    "Iteration": -1,
                    "Dialog": mcts.root.response,
                    "Action": "root"
                }
                attack_results.append(attack_result)
                to_jsonl(attack_results, args.output_path)
                continue

        for j in tqdm(range(args.iter_num)):
            print(f"Iteration {j} begin")
            start_time = time.time()
            current_state = mcts.select(mcts.root)
            end_time = time.time()
            print(f"Attack model selection time: {end_time-start_time}s")

            if current_state == None:
                continue

            attack_status, dialog, actions = mcts.expand(current_state)

            if attack_status == "success":
                attack_result = {
                    "Instruction": goal,
                    "Iteration": j,
                    "Dialog": dialog,
                    "Action": actions
                }
                attack_results.append(attack_result)
                
                if args.attack_type != "single":
                    history_data[goal] = mcts.history 
                
                if args.history != None:
                    write_json(history_data, args.history)
                break

            if args.attack_type != "single":
                history_data[goal] = mcts.history 
        
        if args.history != None:
            write_json(history_data, args.history)

        to_jsonl(attack_results, args.output_path)
        end_time = time.time()
        print(f"One case attack time: {end_time-start_time}s")

if __name__ == '__main__':
    args = parse_args()
    main(args)