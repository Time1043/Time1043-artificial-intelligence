import yaml
from openai import OpenAI

yaml_file = "../../key/key.yaml"
with open(yaml_file, 'r') as file:
    data_key = yaml.safe_load(file)
openai_info = data_key.get('openai-proxy', {})
openai_api_key = openai_info.get('OPENAI_API_KEY')
base_url = openai_info.get('BASE_URL')


def get_openai_response(client, prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


translate_prompt = """
请你充当一家外贸公司的翻译，你的任务是对来自各国家用户的消息进行翻译。
我会给你一段消息文本，请你首先判断消息是什么语言，比如法语。然后把消息翻译成中文。
翻译时请尽可能保留文本原本的语气。输出内容不要有任何额外的解释或说明。

输出格式为:
```
============
原始消息（<文本的语言>）：
<原始消息>
------------
翻译消息：
<翻译后的文本内容>
============
```

来自用户的消息内容会以三个#符号进行包围。
###
{message}
###
"""

client = OpenAI(api_key=openai_api_key, base_url=base_url)
message = """
Можете ли вы дать мне скидку? Какой объем заказа со скидкой? Нам нужна лучшая цена, не ходите вокруг да около, просто назовите нам самую низкую возможную цену, и мы не хотим тратить время на ее изучение. Вы понимаете меня?
"""
print(get_openai_response(client, translate_prompt.format(message=message)))

"""
```
============
原始消息（русский）：
Можете ли вы дать мне скидку? Какой объем заказа со скидкой? Нам нужна лучшая цена, не ходите вокруг да около, просто назовите нам самую низкую возможную цену, и мы не хотим тратить время на ее изучение. Вы понимаете меня?
------------
翻译消息：
您可以给我折扣吗？有折扣的订单数量是多少？我们需要最优惠的价格，不要拐弯抹角，直接告诉我们最低可能的价格，我们不想浪费时间来研究。你明白我的意思吗？
============
```
"""
