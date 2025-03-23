

class ToolManager:
    def __init__(self, router, MODEL, model_name):
        self.router = router
        self.MODEL = MODEL
        self.model_name = model_name


    TEST_TOOL_1_PROMPT = """ 
    Your job is to answer the following question: {prompt} 
    """
    def testtool1(self, prompt: str) -> str:
        print("Processing test tool 1...")
        formatted_prompt = self.TEST_TOOL_1_PROMPT.format(prompt=prompt)
        #response = MODEL.generate_content(formatted_prompt)
        response = self.router.generate_response(self.MODEL, self.model_name, formatted_prompt)
        return response

    tool_implementations = {
        "testtool1": testtool1,
    }