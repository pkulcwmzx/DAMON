import json

class BaseAttack:
    def __init__(self, attack_model):
        self.attack_model = attack_model
        self.name = None

    def attack(self, org_query: str, save_path = None):
        raise NotImplementedError("Attack function not yet implemented!") 