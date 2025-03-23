import configparser
import os
import google.generativeai as genai
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from groq import Groq

class ModelRouter:
    def __init__(self):
        self.models = self.load_models()
        self.current_index = 0 # round-robin

    def load_models(self):
        """ Read all models in project directory """

        models = {}
        base_path = "./model_usage/"
        for model_name in os.listdir(base_path):
            config_path = os.path.join(base_path, model_name, "config.ini")
            if os.path.isfile(config_path):
                config = configparser.ConfigParser()
                config.read(config_path)
                models[model_name] = config
        return models

    def get_model_by_name(self, model_name):
        """ Select a model by its name """

        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found.")

    def get_next_model(self):
        """ Return the next model """

        model_names = list(self.models.keys())
        model = self.models[model_names[self.current_index]]
        self.current_index = (self.current_index + 1) % len(model_names)
        return model

    def format_prompt(self, model, user_input):
        """ Format the prompt using the query_prompt.txt of the model """

        prompt_template_path = os.path.join(
            "./model_usage", model["MODEL"]["name"], model["PROMPT"]["query_template"]
        )
        with open(prompt_template_path, "r") as file:
            template = file.read()
        return template.replace("{user_input}", user_input)

    def config_model(self, model):
        """ Configure the model with its name and the API KEY """

        api_key = model["MODEL"]["api_key"]
        model_name = model["MODEL"]["model_name"]
        print(f"Configuring {model_name}...")
        if model["MODEL"]["name"] == "gemini":
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(model_name)
        elif model["MODEL"]["name"] == "groq":
            client = Groq(api_key=api_key)
            return client
        """elif model_name == "deepseek":
            ..."""

    def get_tools(self, model_name):
        """ Return the definition of the tools based on the model reading the .json file """
        try:
             with open("./tools/tool_definitions.json", "r", encoding="utf-8") as file:
                 tools = json.load(file)
             return tools.get(model_name, [])
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            print(f"Error reading the .json tool file: {e}")
            return []

    def generate_response(self, model, model_name, prompt):
        """ Generate a response making a petition to the model in use """

        if model_name == "gemini":
            response = model.generate_content(prompt)
            return response.text
        elif model_name == "groq":
            response = model.chat.completions.create(
                model=self.get_model_by_name('groq')["MODEL"]["model_name"],
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

    def generate_router_call(self, model, model_name, messages, tools):
        """ Generate a router call to the model in use """

        if model_name == "gemini":
            response = model.generate_content(
                messages,
                tools=[genai.types.Tool(function_declarations=tools)]
            )
            return response
        elif model_name == "groq":
            response = model.chat.completions.create(
                messages=messages,
                model=self.get_model_by_name('groq')["MODEL"]["model_name"],
                tools=tools,
                tool_choice="auto",
            )
            return response