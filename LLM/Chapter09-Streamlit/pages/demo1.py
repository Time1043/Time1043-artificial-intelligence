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
image_path = r"D:\code2\python-code\artificial-intelligence\llm\chapter09-streamlit\data\profile.jpg"
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
