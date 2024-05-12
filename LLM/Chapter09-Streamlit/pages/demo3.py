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
