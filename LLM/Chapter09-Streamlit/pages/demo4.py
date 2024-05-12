import streamlit as st

if "a" not in st.session_state:  # dict
    st.session_state.a = 0  # initialize the value of a in session_state

st.title("Welcome to Streamlit App")

a = 0
clicked = st.button("plus 1")
if clicked:
    st.session_state.a += 1
st.write("The value of a is:", st.session_state.a)
