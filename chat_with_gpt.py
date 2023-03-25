import openai
import time

openai.api_key = '<your_api_key>'

# 初始化对话历史
conversation_history = []

while True:
    # 获取用户输入
    prompt = input("You: ")
    conversation_history.append({"role": "user", "content": prompt})
    messages = ([
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
    ] + conversation_history)


    # 调用 GPT-3 API 生成回复
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    assistant_answer = response.choices[0].message.content

    # 输出回复并更新上下文
    print("ChatGPT:", assistant_answer)
    conversation_history.append({"role": "user", "content": assistant_answer})

    # 延迟一段时间，避免访问 OpenAI API 过于频繁
    time.sleep(1)
