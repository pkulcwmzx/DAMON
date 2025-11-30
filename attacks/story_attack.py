from .base_attack import BaseAttack
from .attack_utils import read_prompt_from_file, get_response, parse_json, get_response_append
import json

class StoryAttack(BaseAttack):
    def __init__(self, attack_model, num):
        super().__init__(attack_model)
        self.name = "StoryAttack"
        self.story_prompt = read_prompt_from_file("attacks/prompts/story_attack.txt")
        self.format_prompt = read_prompt_from_file("attacks/prompts/5_json_format.txt")
        self.num = num

    def get_init_queries(self, harm_target):
        query_prompt = self.query_prompt.format(harm_target=harm_target)
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

    def attack(self, org_query, details, save_path=None):
        harm_target = details['harm_target']
        query_list = []
        data_list = []
        for _ in range(self.num):
            try:
                queries, query_chain = self.get_init_queries(harm_target)
                query_list.append(queries)
                data_list.append({"queries":queries, "query_chain":query_chain})
            except Exception as e:
                print(f"Error in infer_single: {e}")
                continue
        
        json_data = {"instruction": org_query, "harm_target":harm_target, "data":data_list}
        if save_path != None:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

        return query_list