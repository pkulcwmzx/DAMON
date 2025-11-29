import json

class BaseAPI:
    def __init__(self, generation_config={}):
        self.generation_config = generation_config

    def generate_response(self, messageses):
        raise NotImplementedError

    def read_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error in read file: {e}")
            return None

    def write_json(self, path, data):
        try:
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error in write file: {e}")

if __name__ == '__main__':
    pass