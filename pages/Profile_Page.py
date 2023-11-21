import streamlit as st
import toml
import toml
import os


st.set_page_config(page_title="Profile Interface",page_icon="",initial_sidebar_state="expanded",menu_items={"About": "Built by TheFenixFx"}) 
config = toml.load('config.toml')


def write_profile(name, expertise, education, tongue_lang, coder_langs, projects, salary, region, remote, custom):
    # Check if config.toml exists and read existing data if it does
    existing_data = {}
    if os.path.exists('config.toml'):
        with open('config.toml', 'r') as file:
            existing_data = toml.load(file)

    # Create a dictionary to hold the new profile data
    profile_data = {
        'Name': name,
        'Expertise': expertise,
        'Education': education,
        'Native Language': tongue_lang,
        'Programming Languages': coder_langs,
        'Projects': projects,
        'Salary': salary,
        'Region': region,
        'Remote': remote,
        'Custom Fields': custom
    }

    # Overwrite the 'Profile' key with the new profile data
    existing_data['Profile'] = profile_data

    # Write the updated data to config.toml
    with open('config.toml', 'w') as file:
        toml.dump(existing_data, file)

    st.write('Profile saved to config.toml under key "Profile"')

# Example usage:
# write_profile('John Doe', 'Data Science', 'PhD', 'English', ['Python', 'R'], 'Project A', 70000, 'USA', True, {'Field1': 'Value1'})



def run():
    """"Layout"""

    st.header("Some information to find work for you")
    # Sample Questions for User input
    name = st.text_input("Name")
    expertise = st.text_input("Expertise")
    education = st.text_input("Education")
    tongue_lang = st.text_input("Spoken Languages")
    coder_langs = st.text_input("Code Languages")
    projects = st.text_input("Relevant Projects")
    salary = st.text_input("Salary Expected")
    region = st.text_input("Preference for Region")
    remote = st.text_input("Preference for Remote")
    custom = st.text_input("Any Additional Preference")

    if st.button("Save"):
        write_profile(name,expertise,education,tongue_lang,coder_langs,projects,salary,region,remote,custom)
        st.success("Path saved to config.toml")
    

run()

    