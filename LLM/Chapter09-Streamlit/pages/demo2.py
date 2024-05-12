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
