from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatMistralAI(model="mistral-small-2506" , temperature=0.7)
model2 = ChatMistralAI(model="mistral-medium-3.5" , temperature=0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name' : 'Sequential Chain',
    'tag' : ['llm app' , 'report generation' , 'summerization'],
    'metadata' : {'model1' : 'mistral-small-2506' , 'model1_temp' : 0.7 , 'model2' : 'mistral-medium-3.5', 'model2_temp' : 0.5 , 'parser' : 'stroutputparser'}
}

result = chain.invoke({'topic': 'Unemployment in India'} , config=config)

print(result)
