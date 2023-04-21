import openai
import time
from tqdm import tqdm
from multiprocessing import Pool

# openai.api_key = '<your_api_key>'

# 初始化对话历史
conversation_history = []
need_chat_file_list = []

def worker(item):
    # 获取用户输入
    content_into_gpt = item  # 这里去加载出需要输入ChatGPT的内容
    prompt = input("You: ")
    prompt = content_into_gpt
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
    
if __name__ == '__main__':
    items = need_chat_file_list
    with Pool(processes=8) as p:
        with tqdm(total=len(items), desc='total') as pbar:
            for i, _ in enumerate(p.imap_unordered(worker, items)):
                pbar.update()
    print("done")
    
