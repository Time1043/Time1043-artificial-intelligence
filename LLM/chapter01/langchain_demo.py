from typing import List

import yaml
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

yaml_file = "../../key/key.yaml"
with open(yaml_file, 'r') as file:
    data_key = yaml.safe_load(file)
openai_info = data_key.get('openai-proxy', {})
openai_api_key = openai_info.get('OPENAI_API_KEY')
base_url = openai_info.get('BASE_URL')


class BookInfo(BaseModel):
    book_name: str = Field(description="书籍的名字", example="百年孤独")
    author_name: str = Field(description="书籍的作者", example="加西亚·马尔克斯")
    genres: List[str] = Field(description="书籍的体裁", example=["小说", "文学"])


# PydanticOutputParser
output_parser = PydanticOutputParser(pydantic_object=BookInfo)
print(output_parser.get_format_instructions())

prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions} 你输出的结果请使用中文。"),
    ("human", "请你帮我从书籍概述中，提取书名、作者，以及书籍的体裁。书籍概述会被三个#符号包围。\n###{book_introduction}###")
])

book_introduction = """《明朝那些事儿》，作者是当年明月。2006年3月在天涯社区首次发表，2009年3月21日连载完毕，边写作边集结成书出版发行，一共7本。
《明朝那些事儿》主要讲述的是从1344年到1644年这三百年间关于明朝的一些故事。以史料为基础，以年代和具体人物为主线，并加入了小说的笔法，语言幽默风趣。对明朝十六帝和其他王公权贵和小人物的命运进行全景展示，尤其对官场政治、战争、帝王心术着墨最多，并加入对当时政治经济制度、人伦道德的演义。
它以一种网络语言向读者娓娓道出三百多年关于明朝的历史故事、人物。其中原本在历史中陌生、模糊的历史人物在书中一个个变得鲜活起来。《明朝那些事儿》为读者解读历史中的另一面，让历史变成一部活生生的生活故事。
"""
final_prompt = prompt.invoke({
    "book_introduction": book_introduction,
    "parser_instructions": output_parser.get_format_instructions()
})

model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, openai_api_base=base_url)
response = model.invoke(final_prompt)
print(response.content)

result = output_parser.invoke(response)
print(result)
print(result.book_name)
print(result.genres)


"""
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"book_name": {"title": "Book Name", "description": "\u4e66\u7c4d\u7684\u540d\u5b57", "example": "\u767e\u5e74\u5b64\u72ec", "type": "string"}, "author_name": {"title": "Author Name", "description": "\u4e66\u7c4d\u7684\u4f5c\u8005", "example": "\u52a0\u897f\u4e9a\u00b7\u9a6c\u5c14\u514b\u65af", "type": "string"}, "genres": {"title": "Genres", "description": "\u4e66\u7c4d\u7684\u4f53\u88c1", "example": ["\u5c0f\u8bf4", "\u6587\u5b66"], "type": "array", "items": {"type": "string"}}}, "required": ["book_name", "author_name", "genres"]}
```
{
    "book_name": "《明朝那些事儿》",
    "author_name": "当年明月",
    "genres": ["历史", "小说"]
}
"""

"""
book_name='《明朝那些事儿》' author_name='当年明月' genres=['历史', '小说']
《明朝那些事儿》
['历史', '小说']
"""
