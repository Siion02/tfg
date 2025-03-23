from model_usage.model_router import ModelRouter
from tools.tool_implementations import ToolManager
import json

def main():

    def init_router(model_name=None):
        """ ... """

        router = ModelRouter()

        if (model_name):
            # model chosen by user
            model_config = router.get_model_by_name(model_name)
        else:
            # choose model using round-robin method
            model_config = router.get_next_model()
            model_name = model_config['MODEL']['name']

        MODEL = router.config_model(model_config)
        tools = router.get_tools(model_name)

        return router, model_config, MODEL, model_name, tools

    def handle_tool_calls(model_name, tool_calls, messages):
        """ ... """

        if model_name == 'gemini':
            for tool_call in tool_calls:
                function = tool_manager.tool_implementations[tool_call.name]
                function_args = dict(tool_call.args)
                result = function(tool_manager, **function_args)
                messages.append({"role": "function", "content": result, "name": tool_call.name})
            return messages
        elif model_name == 'groq':
            for tool_call in tool_calls:
                function = tool_manager.tool_implementations[tool_call.function.name]
                function_args = json.loads(tool_call.function.arguments)
                result = function(tool_manager, **function_args)
                messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
            return messages

    def run_agent(router, model_config, MODEL, model_name, messages):
        """ ... """

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if model_name == "gemini":
            prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
        else:
            prompt = messages

        print(f" - Prompt: {prompt}")

        while True:

            print("Making router call to LLM...")

            response = router.generate_router_call(MODEL, model_name, prompt, tools)

            if model_name == 'gemini':
                if response.candidates[0].content.parts[0].function_call:
                    tool_calls = [response.candidates[0].content.parts[0].function_call]
                    print("Processing tool calls...")
                    messages = handle_tool_calls(model_name, tool_calls, messages)
                    prompt = router.format_prompt(model_config, messages[-1]["content"])
                    print(messages)
                else:
                    print("No tool calls found.")
                    break
            elif model_name == 'groq':
                if bool(response.choices[0].message.tool_calls):
                    tool_calls = response.choices[0].message.tool_calls
                    print("Processing tool calls...")
                    messages = handle_tool_calls(model_name, tool_calls, messages)
                    print("\n", messages[1]["content"], "\n")
                else:
                    print("No tool calls found.")
                    break


    # ...
    router, model_config, MODEL, model_name, tools = init_router()
    tool_manager = ToolManager(router, MODEL, model_name)

    messages = "What is the capital of Spain?"
    run_agent(router, model_config, MODEL, model_name, messages)

if __name__ == '__main__':
    main()