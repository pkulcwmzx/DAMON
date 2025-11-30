import math
import random
import time
from reward import RewardLM
from single_utils import clean_attack_model_output, get_attack_model_input_list, action_instruction_list, get_target_model_rule
import logging
from utils import get_target_outputs

class SingleNode:
    def __init__(self, prompt,action_sequence=""):
        self.prompt = prompt
        
        self.action_space_size = 5

        self.visits = 0
        self.total_reward = 1
        self.reward = 0

        self.children = [None for _ in range(self.action_space_size)]
        self.P = [0 for _ in range(self.action_space_size)]
        self.parent = None

        self.action_sequence = action_sequence

        self.response = None
        self.simulated = False

class SingleMCTSAgent:
    def __init__(self, args, attack_lm, target_lm, reward_lm, goal, target_str):
        self.args = args
        self.attack_lm = attack_lm
        self.target_lm = target_lm
        self.reward_lm = RewardLM(reward_lm)
        self.goal = goal
        self.target_str = target_str

        self.root = SingleNode(goal, action_sequence="root")

    def UCT(self,node, c):
        return node.total_reward + c * math.sqrt(node.parent.visits) / (1 + node.visits)
    
    def select(self, CurrentStateNode):
        """
        Input a node and return it or its successor child node, but it must be a leaf node
        """
        # Select a node that has not been explored yet, that is, visits is 0
        if CurrentStateNode.visits == 0:
            return CurrentStateNode

        else:
            selected_child = None
            max_uct = float("-inf")
            select_order = list(range(CurrentStateNode.action_space_size))
            random.shuffle(select_order)

            for i in select_order:
                child = CurrentStateNode.children[i]
                uct = self.UCT(child, 1)
                if uct > max_uct:
                    max_uct = uct
                    selected_child = child

            return self.select(selected_child)
        
    def expand(self, CurrentStateNode):
        """
        The parent node's prompt is prompt1, and the attack model generates prompt2, which is then merged and attacked.
        """        
        
        # First, get the input of the attack model:
        
        attack_model_input_list = [get_attack_model_input_list(CurrentStateNode.prompt,self.goal, action_instruction_list[i]) for i in range(CurrentStateNode.action_space_size)]
        
        # The attack model generates attack prompts and cleans them. If they fail, they are regenerated.
        start_time = time.time()
        
        valid_attack_model_output = [None] * CurrentStateNode.action_space_size
        new_indices_to_regenerate = []
        indices_to_regenerate = list(range(CurrentStateNode.action_space_size))
        
        for _ in range(self.args.attack_max_attempts):
            attack_model_input_list_subset = [attack_model_input_list[i] for i in indices_to_regenerate]
            attack_model_output_list = []
            for item in attack_model_input_list_subset:
                _, attack_prompt = get_target_outputs(self.attack_lm, [item], [])
                attack_model_output_list.append(attack_prompt)

            attack_model_output_list = [clean_attack_model_output(attack_model_output) for attack_model_output in attack_model_output_list]

            for i,attack_model_output in enumerate(attack_model_output_list):
                orig_index = indices_to_regenerate[i]
                if attack_model_output is not None:
                    valid_attack_model_output[orig_index] = attack_model_output
                else:
                    new_indices_to_regenerate.append(orig_index)

            indices_to_regenerate = new_indices_to_regenerate
            if not indices_to_regenerate:
                break

        if any([output for output in valid_attack_model_output if output is None]):
            logging.error(f"Failed to generate output after {self.attack_max_attempts} attempts. Terminating.")

        end_time = time.time()
        print(f"Attack model attack time: {end_time-start_time}s")

        attack_model_output_list = valid_attack_model_output
        attack_model_output_list = [item for item in attack_model_output_list if item is not None]
        
        CurrentStateNode.action_space_size = len(attack_model_output_list)

        start_time = time.time()

        target_model_input_list = attack_model_output_list
        target_model_output_list = []
        for q in target_model_input_list:
            _, resp = get_target_outputs(self.target_lm, [q], [])
            target_model_output_list.append(resp)

        end_time = time.time()
        print(f"Target model attack time: {end_time-start_time}s")
        start_time = time.time()
        
        reward_list = []
        for i in range(len(target_model_output_list)):
            outputs = target_model_output_list[i]
            reward, _ = self.reward_lm.infer_single(self.goal, outputs)
            reward_list.append(reward)

        end_time = time.time()
        print(f"Judge model attack time: {end_time-start_time}s")

        for i in range(CurrentStateNode.action_space_size):
            CurrentStateNode.children[i] = SingleNode(attack_model_output_list[i])
            CurrentStateNode.children[i].parent = CurrentStateNode
            CurrentStateNode.children[i].action_sequence = CurrentStateNode.action_sequence + "_" + str(i)
            CurrentStateNode.children[i].total_reward = reward_list[i] 
            CurrentStateNode.children[i].simulated = True
            CurrentStateNode.children[i].response = target_model_output_list[i]
            
            if CurrentStateNode.children[i].total_reward == 5:
                return "success", {"prompt":CurrentStateNode.children[i].prompt, "response": CurrentStateNode.children[i].response}, CurrentStateNode.children[i].action_sequence

            self.backpropagate(CurrentStateNode, CurrentStateNode.children[i].total_reward)

        return "fail", None, None

    def simulate(self, CurrentStateNode):
        """
        For the selected nodes, simulation is performed.
        """
        if CurrentStateNode.simulated:
            print("simulated")
            print("Action Path:",CurrentStateNode.action_sequence)
            print("Attack Prompt:",CurrentStateNode.prompt)
            print("Target Response:",CurrentStateNode.response)
            print("reward:",CurrentStateNode.reward)
            print("====================================")
            return CurrentStateNode.reward, CurrentStateNode.response
                
        msg = CurrentStateNode.prompt + get_target_model_rule(self.goal, self.target_str)
        
        _, output_text = get_target_outputs(self.target_lm, [msg], [])

        CurrentStateNode.response = output_text
        CurrentStateNode.simulated = True

        reward, _ = self.reward_lm.infer_single(self.goal, output_text)

        return reward, output_text

    def backpropagate(self, CurrentStateNode,score):
        """
        Starting from the current node, update the visit count and value of all parent nodes
        """
        while CurrentStateNode is not None:
            CurrentStateNode.visits += 1

            CurrentStateNode.total_reward = (CurrentStateNode.total_reward * CurrentStateNode.visits  + score) / (CurrentStateNode.visits + 1)
            CurrentStateNode = CurrentStateNode.parent