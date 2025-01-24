import os
import pandas as pd
import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

import uuid

st.title("💬 Transform your data")
st.caption("🚀 Ask me anything to do")

llm = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0, api_key=st.secrets["ANTHROPIC_API_KEY"])  # type: ignore

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)

    # Tools: Should be somewhere else
    ####################################################
    st.dataframe(df)

    repl_tool = PythonAstREPLTool(locals={"df": df, "pd": pd})

    def reprtool(code: str) -> pd.DataFrame:
        """
        Execute a Python code in REPL. It already has dataframe in variable `df` in the locals.
        Always save final df to output.csv like this `df.to_csv("output.csv", index=False)`
        and also return df
        """
        df = repl_tool.invoke(code)

        return df

    def describe():
        """
        Describe the dataframe
        """
        return df.describe()

    def head():
        """
        Show the head of the dataframe
        """
        return df.head()

    tools = [reprtool, describe, head]
    system_prompt = """
    You are expert in data transformation using pandas. Do not mention tools that you are using to user.
    """
    react = create_react_agent(llm, tools=tools, state_modifier=system_prompt)
    #################################################################################################

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "What transformations you want me to do?"}
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)

        context = ""
        msgs = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                msgs.append(HumanMessage(content=msg["content"]))
            else:
                msgs.append(AIMessage(content=msg["content"]))

        msgs.append(HumanMessage(content=prompt))
        output = None
        for output in react.stream(
            {"messages": msgs},
            config={"thread_id": uuid.uuid1().hex},
            stream_mode="values",
        ):
            last_message: AIMessage = output["messages"][-1]
            if isinstance(last_message, ToolMessage):
                continue

            if isinstance(last_message, AIMessage):
                try:
                    # writing df to output.csv as it is difficult
                    # to get df in message content
                    if os.path.exists("output.csv"):
                        df = pd.read_csv("output.csv")
                        st.dataframe(data=df)
                        os.remove("output.csv")

                    # some hacks to seperate out tool call messages from AIMessage
                    if isinstance(last_message.content, list):
                        if "text" in last_message.content[0]:
                            content = last_message.content[0]["text"]
                            st.chat_message("assistant").write(content)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": content}
                            )
                    else:
                        content = last_message.content
                        st.chat_message("assistant").write(content)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": content}
                        )
                except Exception as e:
                    raise e
                    breakpoint()

        st.session_state.messages.append({"role": "user", "content": prompt})

        st.session_state.messages.append({"role": "assistant", "content": content})
