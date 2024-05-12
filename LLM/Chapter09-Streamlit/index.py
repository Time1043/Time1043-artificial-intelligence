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
