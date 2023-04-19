import openai
import pickle
import time

openai.api_key = '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'

# 初始化对话历史
conversation_history = []

file = open('all_data_pair.txt', encoding='utf-8')

trues = []
preds = []

prompt = '下面将进行若干次输入，每次输入是一段文本，该文本由若干句子构成，每句都有对应的序号。' \
         '请找出直接表达情感的句子和能解释该情感产生的句子，然后直接输出(a, b)，' \
         '其中a表示情感句对应的序号，b表示原因句对应的序号，不要加任何多余的字词，不要解释。准备好了吗？'
print(prompt)

while True:
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
    preds.append(assistant_answer)

    # 输出回复并更新上下文
    print("ChatGPT:", assistant_answer)
    conversation_history.append({"role": "user", "content": assistant_answer})

    # 延迟一段时间，避免访问 OpenAI API 过于频繁
    time.sleep(1)

    # 获取用户输入
    # prompt = input("You: ")
    line = file.readline().strip().split()
    if not line:
        break
    len = int(line[1])
    true = file.readline().strip()
    true = eval(true)  # <class 'tuple'>
    trues.append(true)
    prompt = ''
    for i in range(len):
        line = file.readline().strip().split(',')
        prompt += line[0] + ',' + line[3] + '\n'
    prompt = prompt.strip()

    print(prompt)

file.close()

with open('trues.pkl', 'wb') as f:
    pickle.dump(trues, f)
with open('preds.pkl', 'wb') as f:
    pickle.dump(preds, f)
