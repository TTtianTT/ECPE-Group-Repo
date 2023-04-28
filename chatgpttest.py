
import openai
import time

openai.api_key = "..."

# 初始化对话历史
conversation_history = []
file = open('all_data_pair.txt', encoding='utf-8')
file_ans = open('test_ans.txt', 'w')
test1 = '请阅读下列对话。请找出含有情感的句子和其对应的能解释其产生原因的句子，请直接输出情感句和原因句所在的位置，不要解释。' \
         '请用列表的形式，比如"[a,b]"的形式进行输出。数字代表的位置为句子在段落中为从前到后的第几句' \
         '其中a表示情感句对应的序号，b表示原因句对应的序号，一定不要加任何多余的字词，不要解释，我只要这个列表。直接输出列表，求你了，千万不要解释'\
         '我将提供给你一些样例输入和输出作为参考，请严格仿照这个输出格式进行输出'
case = 0
prompt = '''
源文本一：
1,null,null,当 我 看到 建议 被 采纳
2,null,null,部委 领导 写给 我 的 回信 时
3,null,null,我 知道 我 正在 为 这个 国家 的 发展 尽着 一份 力量
4,null,null,27 日
5,null,null,河北省 邢台 钢铁 有限公司 的 普通工人 白金 跃
6,null,null,拿 着 历年来 国家 各部委 反馈 给 他 的 感谢信
7,happiness,激动,激动 地 对 中新网 记者 说
8,null,null,27 年来
9,null,null,国家公安部 国家 工商总局 国家科学技术委员会 科技部 卫生部 国家 发展 改革 委员会 等 部委 均 接受 并 采纳 过 的 我 的 建议
目标文本一：
[7,9]
源文本二：

1,null,null,据 白金 跃 介绍
2,null,null,自 1988 年 至今
3,null,null,他 向 国家 各部委 提出 合理化 建议 1000 多条
4,null,null,并 多次 被 各部委 采纳
5,null,null,曾 荣获 国家 十二五 建言献策 个人 一等奖 河北省 建言献策 三等奖
6,null,null,当日
7,null,null,跟 中新网 记者 谈起 建言献策 的 初衷
8,null,null,白金 跃 陷入 回忆
9,happiness,激动,并 略显 激动
目标文本二：
[9.7]

'''
while True:
    # 获取用户输入
    case = case + 1
    test_detail = ''
    line = file.readline()
    while line[1] != ',':
        line = file.readline()
        continue
    while line[1] == ',':
        test_detail += line
        line = file.readline()
    print(test_detail)


    conversation_history.append({"role": "user", "content": test1+prompt+test_detail})
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
    time.sleep(5)

file.close()
file_ans.close()
