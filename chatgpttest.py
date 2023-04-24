# 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
import openai
import time

openai.api_key = "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"

# 初始化对话历史
conversation_history = []
file = open('all_data_pair.txt', encoding='utf-8')
file_ans = open('test_ans.txt', 'w')
test1 = '请阅读下列对话。请找出含有情感的句子和其对应的能解释其产生原因的句子，请直接输出情感句和原因句所在的位置，不要解释。' \
         '请用列表的形式，比如"[a,b]"的形式进行输出。数字代表的位置为句子在段落中为从前到后的第几句' \
         '其中a表示情感句对应的序号，b表示原因句对应的序号，一定不要加任何多余的字词，不要解释，我只要这个列表。直接输出列表，求你了，千万不要解释'
case = 0
while True:
    # 获取用户输入
    test_detail = ''
    line = file.readline()
    while line[1] != ',':
        line = file.readline()
        continue
    while line[1] == ',':
        test_detail += line
        line = file.readline()
    print(test_detail)


    conversation_history.append({"role": "user", "content": test1+test_detail})
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
    file_ans.write(f'case:{case}\n')
    file_ans.write(assistant_answer)
    # 延迟一段时间，避免访问 OpenAI API 过于频繁
    time.sleep(10)
file.close()
file_ans.close()
