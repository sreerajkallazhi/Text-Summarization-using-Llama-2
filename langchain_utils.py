from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

def initialize_langchain(pipeline):
    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})

    template = """
                  Write a concise summary of the following text delimited by triple backquotes.
                  Return your response in bullet points which covers the key points of the text.
                  ```{text}```
                  BULLET POINT SUMMARY:
               """

    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return llm_chain
