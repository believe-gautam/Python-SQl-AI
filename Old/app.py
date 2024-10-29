import chainlit as cl
from query_engine import setup_chat_engine

@cl.on_chat_start
async def start_chat():
    chat_engine = setup_chat_engine()
    cl.user_session.set("chat_engine", chat_engine)

@cl.on_message
async def handle_message(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine")
    response = await cl.make_async(chat_engine.query)(message.content)

    response_message = cl.Message(content="")
    for token in response.response:
        await response_message.stream_token(token=token)
    await response_message.send()

# Run this via terminal with: chainlit run app.py
