import json

from openai import OpenAI
import yaml

yaml_file = "../../key/key.yaml"
with open(yaml_file, 'r') as file:
    data_key = yaml.safe_load(file)
openai_info = data_key.get('openai-proxy', {})
openai_api_key = openai_info.get('OPENAI_API_KEY')
base_url = openai_info.get('BASE_URL')

client = OpenAI(api_key=openai_api_key, base_url=base_url)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt},
    ]
)
content = response.choices[0].message.content
result = json.loads(content)
print(result)
print(result[0]["phone"])

"""
[
    {'order_id': '001', 'customer_name': 'Alice', 'order_item': 'iPhone 12', 'phone': '123-456-7890'},
    {'order_id': '002', 'customer_name': 'Bob', 'order_item': 'Samsung Galaxy S21', 'phone': '987-654-3210'},
    {'order_id': '003', 'customer_name': 'Charlie', 'order_item': 'Google Pixel 5', 'phone': '456-789-0123'}
]

123-456-7890
"""
