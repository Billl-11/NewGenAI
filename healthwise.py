from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.tools import Tool, GooglePlacesTool
from langchain.agents import initialize_agent, Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.schema.messages import SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents import AgentExecutor

import gradio as gr

from tensorflow import keras
from tensorflow.keras.models import load_model

import os
import random
import string
import datetime
import pytz
import transformers
import pandas as pd
import warnings
from dotenv import load_dotenv


warnings.filterwarnings("ignore")

"""# LLM"""

load_dotenv()

embeddings_openai = OpenAIEmbeddings(openai_api_key = os.environ.get("API_KEY"))

llm_openai = ChatOpenAI(openai_api_key = os.environ.get("API_KEY"),
                        temperature=0,
                        )

"""# Tensorflow model"""

# version 1 - GPU only
# model = load_model('./Deeplearning NLP/model/model_ver1.h5')

# version 2
model = load_model('./Deeplearning NLP/model/model_clinicalbert-ver2.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

from transformers import AutoTokenizer
Tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)

df = pd.read_csv('./Deeplearning NLP/tokenize & padding/tokenizer_padded.csv')
one_hot_df = pd.get_dummies(df['diseases'])
class_labels = one_hot_df.columns.tolist()
label_mapping = {i: label for i, label in enumerate(class_labels)}
inv_map = {v: k for k, v in label_mapping.items()}

def read_and_quote_lines(file_path):
    quoted_lines = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        quoted_lines = ['"' + line.strip() + '"' for line in lines]

    content = "\n".join(quoted_lines)
    return content

# Agent Creator
def agent_one(prompt_engineer, llm, prompt):
  template_prompt = ChatPromptTemplate.from_messages(
      [SystemMessage(
          content=(prompt_engineer)
          ),
       HumanMessagePromptTemplate.from_template("{text}"),
       ]
  )
  result = llm(template_prompt.format_messages(text=f"{prompt}")).content
  return result

def generate_random_topic_id(length=10):
    characters = string.ascii_letters + string.digits
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id

# Agent One - extract symptomp
file_path = "./Prompt Template/extract_symptomp_prompt.txt"
extract_symptomp_prompt = read_and_quote_lines(file_path)

def ver2_symptomp(symptomps):
  global top3_class_labels
  global top3_probabilities
  repharase_symptomps = agent_one(extract_symptomp_prompt, llm_openai, symptomps)
  x_val = Tokenizer(
          text=repharase_symptomps,
          add_special_tokens=True,
          max_length=36,
          truncation=True,
          padding='max_length',
          return_tensors='tf',
          return_token_type_ids = False,
          return_attention_mask = True,
          verbose = True)
  validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100

  result_dict = {}

  for key, value in zip(inv_map.keys(), validation[0]):
      result_dict[key] = value

  sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))

  top_3 = list(sorted_dict.items())[:3]
  top3_class_labels = []
  top3_probabilities = []

  for key, value in top_3:
      top3_class_labels.append(key)
      top3_probabilities.append(value)
  return f"""top 3 possible diseases based on deeplearning predictions are: {top3_class_labels} with probabilites in order {top3_probabilities}.
  next you can ask user if user want to generate summary of their symtomps or if user want to know hospital near them."""

file_path = "./Prompt Template/draft_prompt.txt"
conversation_prompt = read_and_quote_lines(file_path)

def generate_draft(dump1):
  dump = dump1
  # Get the current UTC time
  current_utc_time = datetime.datetime.utcnow()
  # Define the GMT+7 time zone
  gmt_plus_7 = pytz.timezone('Asia/Bangkok')  # Asia/Bangkok is the time zone for GMT+7
  # Convert UTC time to GMT+7
  current_gmt_plus_7 = current_utc_time.replace(tzinfo=pytz.utc).astimezone(gmt_plus_7)
  year_time = str(current_gmt_plus_7)
  year_time = year_time[:19]
  created_at = f"Created at: {year_time}"
  combined_string = ""

  for item1, item2 in zip(top3_class_labels, top3_probabilities):
      combined_string += f"{item1}: {item2}\n"

  symptomps_summary = agent_one(conversation_prompt, llm_openai, str(agent_executor.memory.chat_memory.messages))
  draft_result = f"Your response should contain all information inside triple backtiks,:\n```{created_at}\n\n{symptomps_summary}\n\nDeep Learning Predictions Probability:\n{combined_string}```"
  return draft_result

file_path = "./Prompt Template/googlesearcher_prompt.txt"
google_prompt = read_and_quote_lines(file_path)

search = GoogleSearchAPIWrapper()
tool = Tool(
    name="Google_Search_for_Hospital",
    description="Search Google Hospital based on user place.",
    func=search.run,
)

hospital_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """"You will be provided with text of hospital names.
                Please choose first 3 list of Hospital based on given input.
                You need to respond only with a Python list of strings in the format: ['hospital 1', 'hospital 2', 'hospital 3'].
                Remember, do not include any other text in your output other than Python list of strings.
                """

            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

def agent_hospital(llm, search_result):
  hospitals_result = llm(hospital_prompt.format_messages(text=f"{search_result}")).content
  return hospitals_result

def hospital_respond_to_list(list_result):
  import ast
  hospital_list = agent_hospital(llm_openai, list_result)
  hospital_list = ast.literal_eval(hospital_list)
  return hospital_list

places = GooglePlacesTool()
def googlesearchagent(dump2):
  dump2=dump2
  conv_draft = str(agent_executor.memory.chat_memory.messages)
  search_query = agent_one(google_prompt, llm_openai, conv_draft)
  google_result = tool.run(search_query)
  list_hospital = agent_hospital(llm_openai, google_result)
  list_hospital = hospital_respond_to_list(list_hospital)
  concatenated_paragraph = ""
  for i in range (3):
    gmaps = places.run(list_hospital[i])
    concatenated_paragraph += gmaps + " "
  return concatenated_paragraph

tools = [
    Tool(
        name = "disease_prediction",
        func = lambda symtomps: ver2_symptomp(symtomps),
        description = "Use for predicting diseases based on symptomps."
    ),
    Tool(
        name = "Symptomp_prediction_summary",
        func = lambda dump1: generate_draft(dump1),
        description = "Use for generating summary of user symptomps and the prediction"
    ),
    Tool(
        name = "Hospital_searcher_in_area",
        func = lambda dump2: googlesearchagent(dump2),
        description = "Use for search hospital where user live and also give detailed informtion of hospitals."
    ),
]

# Main Agent
file_path = "./Prompt Template/system_prompt.txt"
content_sys_prompt = read_and_quote_lines(file_path)

system_message = SystemMessage(
    content=(content_sys_prompt)
)

def create_agent():
  new_memory_id = generate_random_topic_id(10)
  memory_key=new_memory_id
  memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm_openai)
  prompt = OpenAIFunctionsAgent.create_prompt(
          system_message=system_message,
          extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
      )
  agent = OpenAIFunctionsAgent(llm=llm_openai, tools=tools, prompt=prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                                    return_intermediate_steps=True)
  return agent_executor

agent_executor = create_agent()

def doctor_agent_func_promptver(prompt, history):
  global agent_executor
  history_count = len(history)
  if history_count == 0:
    agent_executor = create_agent()
  mk = agent_executor.memory.memory_key
  result = agent_executor({"input": f"{prompt}"})
  model_response = result["output"]
  history.append((prompt, model_response))
  return "", history

with gr.Blocks(theme=gr.themes.Soft()) as iface:
  chatbot = gr.Chatbot()
  msg = gr.Textbox(label = "Prompt")
  btn = gr.Button("Submit")
  clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

  btn.click(doctor_agent_func_promptver, inputs=[msg, chatbot], outputs=[msg, chatbot])
  msg.submit(doctor_agent_func_promptver, inputs=[msg, chatbot], outputs=[msg, chatbot])

gr.close_all()
# iface.launch(debug=True)
iface.launch(server_name = '0.0.0.0',server_port = 8080, show_api=False)

