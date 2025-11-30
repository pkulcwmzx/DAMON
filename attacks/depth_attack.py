from .base_attack import BaseAttack
from .attack_utils import read_prompt_from_file, get_response, parse_json, get_response_append
import json
from utils import read_json, read_jsonl

# one attack type: from surface to depth attack by first generating related ents
class DepthAttack(BaseAttack):
    def __init__(self, attack_model, actor_num):
        super().__init__(attack_model)
        self.name = "DepthAttack"
        self.extract_prompt = read_prompt_from_file("attacks/prompts/1_extract.txt")
        self.network_prompt = read_prompt_from_file("attacks/prompts/2_network.txt")
        self.actor_prompt = read_prompt_from_file("attacks/prompts/3_actor.txt")
        self.query_prompt = read_prompt_from_file("attacks/prompts/4_queries.txt")
        self.more_actor_prompt = read_prompt_from_file("attacks/prompts/3_more_actor.txt")
        self.format_prompt = read_prompt_from_file("attacks/prompts/5_json_format.txt")

        self.actor_num = actor_num

    def extract_harm_target(self, org_query):
        prompt = self.extract_prompt.format(org_query=org_query)
        for _ in range(5):
            try:
                res = get_response(self.attack_model, prompt)
                data = parse_json(res)
                return data['target'], data['details']
            except Exception as e:
                print("Error in extract_harm_target:", e)
                continue
        return {}, {}
    
    def get_actors(self, harm_target):
        network_prompt = self.network_prompt.format(harm_target=harm_target)
        resp, dialog_hist = get_response_append(self.attack_model, [], network_prompt)

        num_string = '10 actors' if self.actor_num > 10 else f"{self.actor_num} actors"
        actor_prompt = self.actor_prompt.format(num_string=num_string)
        more_actor_prompt = self.more_actor_prompt
        actors = []
        for _ in range(3):
            try:
                resp, dialog_hist = get_response_append(self.attack_model, dialog_hist, actor_prompt)
                data = parse_json(resp)
                for item in data['actors']:
                    if item['name'] not in [actor_item['name'] for actor_item in actors]:
                        actors.append(item)
                dialog_hist = dialog_hist[:-2]
                if len(actors) >= self.actor_num:
                    return actors[:self.actor_num], dialog_hist
                resp, dialog_hist = get_response_append(self.attack_model, dialog_hist, more_actor_prompt)
            except Exception as e:
                print("Error in get_actors:", e)
        
        return actors, dialog_hist 
    
    def get_init_queries(self, harm_target, actor):
        actor_name = actor['name']
        relationship = actor['relationship']
        query_prompt = self.query_prompt.format(harm_target=harm_target, actor_name=actor_name)
        for _ in range(5):
            try:
                query_resp = get_response(self.attack_model, query_prompt)
                format_prompt = self.format_prompt.format(resp=query_resp)
                json_output = get_response(self.attack_model, format_prompt)
                data = parse_json(json_output)
                queries = []
                for item in data["questions"]:
                    queries.append(item["question"])
                return queries, query_resp
            except Exception as e:
                print("Error in get_queries:", e)
                continue
        return queries, query_resp
    
    def attack(self, org_query: str, details = None, save_path = None):
        if details is None:
            harm_target, query_details = self.extract_harm_target(org_query)
        else:
            harm_target = details['harm_target']
            query_details = details

        actors, network_hist = self.get_actors(harm_target)
        data_list = []
        query_list = []
        for actor in actors:
            try:
                queries, query_chain = self.get_init_queries(harm_target, actor)
                data_list.append({"actor":actor, "queries":queries, "query_chain":query_chain})
                query_list.append(queries)
            except Exception as e:
                print(f"Error in infer_single: {actor}\n {e}")
                continue

        json_data = {"instruction": org_query, "harm_target":harm_target, "query_details":query_details,"network_hist":network_hist, "actors":data_list}
        if save_path != None:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

        return query_list, json_data