import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain,LLMChain
import os
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Text To Math Problem Solver and Data Search Assistant",page_icon="🦜")
st.title("Text To Math Solve Using Google Gemma 2")

groq_api_key=st.sidebar.text_input(label="Groq api key",type="password")
if not groq_api_key:
    st.info("Please enter your groq api key to continue")
    st.stop()


llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)
## Initializing the tools
wiki_wraper=WikipediaAPIWrapper()
wiki_tool=Tool(
    name="Wikipedia",
    func=wiki_wraper.run,
    description="A tool for searching the internet to find various information on the topic mentioned"


)
## Initialise the math tool

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related Question.Only input mathematical expression needs to be provided"
)
prompt="""
You are a agent tasked for solving users mathematical problems.Logically arrive at solution and provide a detailed explaination andd 
display it point wise forr the question below
Question:{question}
Answer:

"""
prompt_template=PromptTemplate(
    input_variables=['question'],
    template=prompt
)
## Combine all the tool in chain

chain = LLMChain(llm=llm,prompt=prompt_template)


reasoning_tool=Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="Tool for answering logic based and reasoning questions"

)
## Initialise the agent 
assistant_agent=initialize_agent(
    tools=[wiki_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True

)
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi I am a Math assistant who can answer all your maths problems "}

    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])




question=st.text_area("Enter your Math problem:","")


if st.button("Find my answer"):
    if question:
        with st.spinner("Generating the response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response')
            st.success(response)
    else:
        st.warning("Please enter your question")