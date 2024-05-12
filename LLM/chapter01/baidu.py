import erniebot

response = erniebot.ChatCompletion.create(
    model="ernie-3.5",
    messages=[
        {
            "role": "user",
            "content": "你好，请介绍一下你自己。"
        }
    ]
)
print(response.get_result())
