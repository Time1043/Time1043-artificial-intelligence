# Large Language Model

- 定位

  精简





# LLM (linlili)

- Reference

  [openai api docs](https://platform.openai.com/docs/introduction), [openai-playground](https://platform.openai.com/playground)

- Reference
  
  kimi, 

  spark, 

  baidu, 

  tongyi, 
  
- Reference - course

  [linlili 课程代码](https://n6fo0mbcz6.feishu.cn/drive/folder/ZEpgfI7OiloJaKdf8IIc6eg5nnd)

  



- 今非昔比

  过去：苦读机器学习算法、从头训练；代码调整模型

  现在：大模型在理解及生成自然语言上极大提升、大模型API；允许自然语言的要求

- BigPicture

  AI模型本身：无法记忆历史对话、不会阅读外部文档、不擅长数学计算不懂如何上网

  用代码武装：给模型添加**记忆**、给模型读取**外部知识库**的能力、通过推理协同让模型能够根据任务要求自主调用一系列**外部工具**

  通用框架：国内国外、云端本地、开源闭源 (迁移)

  ![](res/Snipaste_2024-04-10_21-40-06.png)





## API基础 (openai)

- 用代码与AI对话

  API基础：密钥、请求、API计费(token、tiktoken)
  
  API参数：`max_tokens`、`temperature`、`置信度阈值`、`存在惩罚`、`频率惩罚` (长度 创造性 随机性)
  
  用法提示：文本总结、文本撰写、文本分类、文本翻译 





### 入门调用

- 快速入门 (代理api_key 关掉梯子)

  环境

  ```bash
  pip install openai  # 请求响应
  pip install tiktoken  # token计算
  
  ```

  编码

  创建实例 `client = OpenAI()`

  调用方法 `client.chat.completions.create(model, message)`

  ```python
  from openai import OpenAI
  import yaml
  
  # 保护密钥信息
  yaml_file = "../../key/key.yaml"
  with open(yaml_file, 'r') as file:
      data_key = yaml.safe_load(file)
  openai_info = data_key.get('openai-proxy', {})
  openai_api_key = openai_info.get('OPENAI_API_KEY')
  base_url = openai_info.get('BASE_URL')
  # print(openai_api_key)
  # print(base_url)
  
  # 创建OpenAI客户端实例 发送请求得到响应
  client = OpenAI(api_key=openai_api_key, base_url=base_url)
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "你是一个热爱好故事的文学家，同时也是有着深厚的文学功底的作家。"},
          {"role": "user", "content": "给我推荐一个影视作品吧"},
          {"role": "assistant", "content": "好的，我将给您推荐《doctor who》，并且我有足够打动你的理由！我最喜欢其中的一句话，他有着直击灵魂的力量，相信这也能鼓动你！"},
          {"role": "user", "content": "是吗？我很期待你的理由和你喜欢的那句话！"},
      ],
      max_tokens=300
  )
  print(response)
  
  """
  ChatCompletion(
      id='chatcmpl-9CRfLldAmmg5MR4jtbWdKBs0yq8so', 
      choices=[
          Choice(
              finish_reason='stop', 
              index=0, 
              logprobs=None, 
              message=ChatCompletionMessage(
                  content='当然！《doctor who》是一部英国长寿科幻电视剧，讲述一位外貌年轻但实际年龄几百岁的时间领主——博士（Doctor）的冒险故事。他是一名拥有时间旅行能力的时间领主，驾驶着TARDIS（时间飞船）在时间和空间中进行奇幻的冒险。这部剧探讨了关于友谊、勇气、悲伤、希望等人性主题，同时也引发观众对生命、宇宙的深刻思考。\n\n我最喜欢的一句话来自第十一任博士，他说：“我们不只是感受着时间的流逝，我们也是时间的流逝。我们是一种能够感知时间的存在，我们是时间本身。”这句话表达了博士这个角色对时间和生命的独特理解，深深地触动了我。\n\n相信看完《doctor who》，您也会被这部作品中深刻的情感和深度的思考所感动！希望您会喜欢！', 
                  role='assistant', 
                  function_call=None, 
                  tool_calls=None
              )
          )
      ], 
      created=1712753311, 
      model='gpt-3.5-turbo-0125', 
      object='chat.completion', 
      system_fingerprint='fp_b28b39ffa8', 
      usage=CompletionUsage(completion_tokens=334, prompt_tokens=122, total_tokens=456)
  )
  """
  
  ```

- token计费

  [web tokenizer](https://platform.openai.com/tokenizer)、tiktoken lib `len(encoding.encode(text))`

  ```python
  import tiktoken
  
  text = """当然！《doctor who》是一部英国长寿科幻电视剧，讲述一位外貌年轻但实际年龄几百岁的时间领主——博士（Doctor）的冒险故事。
  他是一名拥有时间旅行能力的时间领主，驾驶着TARDIS（时间飞船）在时间和空间中进行奇幻的冒险。这部剧探讨了关于友谊、勇气、悲伤、希望等人性主题，
  同时也引发观众对生命、宇宙的深刻思考。\n\n
  我最喜欢的一句话来自第十一任博士，他说：“我们不只是感受着时间的流逝，我们也是时间的流逝。我们是一种能够感知时间的存在，我们是时间本身。”
  这句话表达了博士这个角色对时间和生命的独特理解，深深地触动了我。\n\n
  相信看完《doctor who》，您也会被这部作品中深刻的情感和深度的思考所感动！希望您会喜欢！', 
  """
  
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 返回对应编码器
  print(encoding.encode(text))  # 返回token id组成的列表 [40265, 61994, 6447, 28038, 38083, 889,
  print(len(encoding.encode(text)))  # 计算token
  
  ```

  ContextWindow

  ![](res/Snipaste_2024-04-10_21-38-39.png)





### 参数设置

- 调参 (长度 创造性 随机性)

  `max_tokens`：强硬控制，不会据此调节长度，而是拦腰截断 -> `回复在500字内...`

  `temperature`：随机性创造性，0到2之间默认为1，越低随机性越低 (太高甚至不按人类语言)

  `top_p`：控制回答的随机性和创造性、0到1之间 (一般不要同时修改)

  - temperatue：改变各个token的概率分布：温度越低，概率分布的峰高，概率较高的词选择权重增大、概率较低的词较容易忽略，模型的输出具有确定性
  - top_p：不改变词的概率分布，而是关注于截取概率分布的一个子集，子集的累积概率大于等于top_p
  
  ![](res/Snipaste_2024-04-10_21-27-43.png)
  
  
  
  `frequency_penalty`：多大程度上惩罚重复内容、-2到2之间默认为0、**出现得越频繁**今后生成的概率降低 (想要减少高频词出现次数)
  
  `presence_penalty`：降低文本的重复性、-2到2之间默认为0、出现了就**同等情况降低频率** (想要重复词少)
  
  ![](res/Snipaste_2024-04-10_21-44-51.png)
  
  



### prompt engineering

- 提示词工程 [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)

  研究如何提高和AI的沟通质量和效率，核心是提示的开发和优化
  
  规范、格式、零样本小样本、思维链和分步骤思考



- 提示词最佳实践

  限定输出格式

  零样本和小样本

  思维链与分步骤思考

- 提示工程原则

  1. 使用最新的模型
  2. 指令放在提示的开头，用`###`或`"""`分割指令和上下文
  3. 尽可能对上下文和输出的长度、格式、风格等给出具体、描述性、详细的要求
  4. 通过一些例子来阐明想要的输出格式
  5. 先从零样本提示开始，效果不好则用小样本提示
  6. 减少空洞和不严谨的描述
  7. 与其告知不应该做什么，不如告知应该做什么

  ![Snipaste_2024-04-10_21-58-50](res/Snipaste_2024-04-10_21-58-50.png)

- 为了后续

  限定输出格式：yaml、xml、json (不要包含任何没必要的补充信息) 

  小样本提示：`user`、`assistant`

  思维链：[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

  - 算数、常识、符号推理等复杂任务。`let's think step by step.`

- 限定输出格式

  ```python
  # 限定输出格式
  response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    response_format={ "type": "json_object" },
    messages=[...]
  )
  
  ```

  ```python
  import json
  
  from openai import OpenAI
  import yaml
  
  yaml_file = "../../key/key.yaml"
  with open(yaml_file, 'r') as file:
      data_key = yaml.safe_load(file)
  openai_info = data_key.get('openai-proxy', {})
  openai_api_key = openai_info.get('OPENAI_API_KEY')
  base_url = openai_info.get('BASE_URL')
  
  prompt = """生成一个由三个虚构的订单信息所组成的列表，以SON格式进行返回。
  JSON列表里的每个元素包含以下信息：
  order_id、customer_name、order_item、phone。
  所有信息都是字符串。
  除了JSON之外，不要输出任何额外的文本。"""
  
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
  print(result[0]["phone"])  # 可以直接被代码解析
  
  """
  [
      {'order_id': '001', 'customer_name': 'Alice', 'order_item': 'iPhone 12', 'phone': '123-456-7890'},
      {'order_id': '002', 'customer_name': 'Bob', 'order_item': 'Samsung Galaxy S21', 'phone': '987-654-3210'},
      {'order_id': '003', 'customer_name': 'Charlie', 'order_item': 'Google Pixel 5', 'phone': '456-789-0123'}
  ]
  
  123-456-7890
  """
  
  ```

- 小样本提示 (零样本提示 即直接丢问题给AI 没有给任何示范)

  ![](res/Snipaste_2024-04-10_22-19-23.png)

  







- 应用

  文本总结、文本撰写、文本分类、文本翻译





## API基础 (spark)







## API基础 (kimi)









## LangChain

- LangChain

  模块、model IO、











## Streamlit

- 网站开发

  复杂技术栈：html, css, js, ts; [bootstrap](https://v3.bootcss.com/css/), nodejs, vue, react; spring, django, flask

  简单实现：[streamlit (前端框架 + 后端框架 + 云服务器)](https://streamlit.io/) 



- 准备环境

  ```bash
  pip install streamlit
  streamlit hello
  
  ```

  

- 总览

  添加文本图片表格

  添加输入组件

  调整网站布局和增强容器

  管理用户会话状态

  创建多页网站

  部署应用

- 特性

  streamlit在两种情况下会重新运行整个py文件 (对源代码修改 用户与组件交互)





### 基础组件

- 各个组件

  添加文本图片表格

  ```python
  import streamlit as st
  import pandas as pd
  
  """
  cmd: streamlit run page1.py
  """
  
  # show text
  st.title("Streamlit App 😉")
  st.write("### Welcome to the Streamlit App")  # string md
  
  # show variable
  variable = 8080 * 4
  variable
  # show list
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  # show dictionary
  {"name": "John", "age": 30, "city": "New York"}
  
  # show image
  image_path = r"D:\code2\python-code\artificial-intelligence\LLM\Chapter09-Streamlit\data\profile.jpg"
  st.image(image_path, width=200)
  
  # show table
  df = pd.DataFrame(
      {
          "Name": ["John", "Jane", "Bob", "Alice", "Tom"],
          "Age": [30, 25, 40, 35, 28],
          "City": ["New York", "Paris", "London", "Berlin", "Tokyo"],
          "Graduated": ["CMU", "Harvard", "Stanford", "MIT", "Yale"],
          "Gender": ["Male", "Female", "Male", "Female", "Male"],
      }
  )
  st.dataframe(df)  # interactive table
  st.divider()  # horizontal line
  st.table(df)  # static table
  
  ```

  添加输入组件

  文字输入、数字输入、勾选框、按钮

  ```python
  import streamlit as st
  
  st.title("Welcome to Streamlit App")
  
  # Input text
  name = st.text_input("Enter your name: ")
  password = st.text_input("Enter a keyword: ", type="password")
  
  # Input large text
  paragraph = st.text_area("Please enter a paragraph about yourself: ")
  
  if name and paragraph:
      st.write(f"Hello {name}! Welcome! I've gotten to know you: ")
      st.write(paragraph)
  
  # Input number
  st.divider()
  age = st.number_input(
      "Enter your age: ",
      min_value=8, max_value=150, value=25, step=2
  )
  st.write(f"Your age is {age} years old.")
  
  # Checkbox
  st.divider()
  checked = st.checkbox("I agree to the terms and conditions")
  if checked:
      st.write("Thank you for agreeing to the terms and conditions.")
  
  # Button
  submit = st.button("Submit")
  if submit:
      st.write("Form submitted successfully!")
  
  ```

  单选按钮、单选框、多选框、滑块、文件上传器

  ```python
  import streamlit as st
  
  st.title("Welcome to the Streamlit App")
  
  # Radio Button
  st.divider()
  gender = st.radio(
      "What is your gender?",
      ["Male", "Female", "Other"],
      index=0
  )
  if gender == "Male":
      st.write("Welcome, Mr. Smith!")
  elif gender == "Female":
      st.write("Welcome, Ms. Smith!")
  else:
      st.write("Welcome for you!")
  
  # select box
  st.divider()
  contact = st.selectbox(
      "Select your contact method",
      ["Email", "Phone", "Facebook"]
  )
  if contact:
      st.write(f"All right, we will contact you via {contact}")
  
  # multi select
  st.divider()
  interests = st.multiselect(
      "What are your interests?",
      ["Reading", "Hiking", "Traveling", "Cooking"]
  )
  if interests:
      st.write(f"You are interested in {', '.join(interests)}")
  
  # slider
  st.divider()
  height = st.slider(
      "What is your height (cm)?",
      min_value=80, max_value=250, value=170, step=3
  )
  if height:
      st.write(f"You are {height} cm tall")
  
  # file uploader
  st.divider()
  upload_file = st.file_uploader(
      "Please upload your resume (only: pdf, md, txt, py):",
      type=["pdf", "md", "txt", "py"]
  )
  if upload_file:
      st.write(f"Thank you for uploading {upload_file.name}.")
      st.write(f"Preview file content: {upload_file.read()}")
  
  ```

  调整网站布局和增强容器

  侧边栏、分列、选项卡、折叠展开

  ```python
  import streamlit as st
  
  # sidebar
  with st.sidebar:
      name = st.text_input("Please enter your name")
      gender = st.radio(
          "Please select your gender",
          ["Secret", "Male", "Female"],
          index=0
      )
  
  if gender == "Male":
      st.title(f"Welcome Mr. {name}! ")
  elif gender == "Female":
      st.title(f"Welcome Mrs. {name}! ")
  else:
      st.title(f"Welcome {name}! ")
  
  # multi-columns
  column1, column2 = st.columns([3, 4])
  with column1:
      st.divider()
      age = st.number_input(
          "Please enter your age",
          min_value=8, max_value=150, value=25, step=1
      )
      st.divider()
      height = st.slider(
          "What is your height (cm)?",
          min_value=80, max_value=250, value=170, step=3
      )
      st.divider()
      interests = st.multiselect(
          "Please select your interests",
          ["Reading", "Hiking", "Traveling", "Cooking"]
      )
  with column2:
      paragraph = st.text_area(
          "Please enter a paragraph about yourself: ",
          height=480,
          value="I am a software engineer with a passion for data science and machine learning..."
      )
  
  # multi-tabs
  tab1, tab2, tab3 = st.tabs(["Movie", "Music", "Sports"])
  with tab1:
      movie = st.multiselect(
          "What is your favorite movie genre?",
          ["Action", "Comedy", "Drama", "Sci-fi"]
      )
  with tab2:
      music = st.multiselect(
          "What is your favorite music genre?",
          ["Pop", "Rock", "Hip-hop", "Jazz"]
      )
  with tab3:
      sport = st.multiselect(
          "What is your favorite sport?",
          ["Basketball", "Football", "Baseball", "Tennis"]
      )
  
  # expander
  with st.expander("Contact Information"):
      email = st.text_input("Please enter your email")
      phone = st.text_input("Please enter your phone number")
      address = st.text_input("Please enter your address")
  
  # check
  st.divider()
  checked = st.checkbox("I agree to the terms and conditions")
  if checked:
      st.write("Thank you for agreeing to the terms and conditions.")
  else:
      st.write("Please agree to the terms and conditions to continue.")
  
  # submit button
  if st.button("Submit"):
      st.success("Thank you for submitting the form!")
  
  ```

  



### 会话和多页面

- 会话状态存储值 (不关闭浏览器的标签页)

  ```python
  import streamlit as st
  
  if "a" not in st.session_state:  # dict
      st.session_state.a = 0  # initialize the value of a in session_state
  
  st.title("Welcome to Streamlit App")
  
  a = 0
  clicked = st.button("plus 1")
  if clicked:
      st.session_state.a += 1
  st.write("The value of a is:", st.session_state.a)
  
  ```

- 多页面

  ```
  ls -R
  .:
  data  index.py  pages
  ./pages:
  demo1.py  demo2.py  demo3.py  demo4.py
  
  
  streamlit run index.py
  
  ```

  



### 社区部署

- 部署

  localhost

  公网ip

  

- Streamlit部署流程简单

  ```bash
  pip freeze > requirements.txt
  
  # push github
  
  ```

  [streamlit](https://share.streamlit.io/)

  

  





































