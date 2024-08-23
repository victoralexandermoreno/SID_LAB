#!/usr/bin/env python
# coding: utf-8

# In[8]:


# import streamlit as st


# st.title("SID Lab: Thematic Analysis Tool")
# st.header("Social Identity in Dialogue Lab | Professor Kiara Sanchez | Summer Term 2024")


# uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# if uploaded_file is not None:
#     st.write("File uploaded successfully!")


# In[18]:


import openai
import random 
import pandas as pd
import numpy as np

openai.api_key = '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'


# In[12]:


prompt_template = """

Instructions:

You will receive a list of themes along with their definitions and examples, as well as a CSV file.
The CSV file contains a column labeled "Text" and another column labeled "Label" 
Your task is to analyze each entry in the "Text" column and determine whether the specified themes are present.

Texts labeled as Benefits should be searched for ["Awareness of Racial Issues and Empathy", "Trust & Solidarity"] ONLY. 
Texts labeled as Obstacles should be searched for ["Discomfort", "Bias/racism", "Communication Break-down", "Ignorance", "Lack of Interest", "No Obstacles"] ONLY. 
Texts labeled as Overcomes should be searched for ["Active Listening", "Education", "Communicating about Challenges", "Acceptance, "Avoidance"] ONLY. 



Steps:

1. Review Themes: Familiarize yourself with the provided definitions and examples of each theme. These definitions are intended to guide your analysis; the exact wording does not need to be present in the text.

2. Analyze Text Entries: For each text entry, determine the presence of the specified themes. Multiple themes can be present in a single entry.

3. Coding:
    - If a theme is present, mark it with a "1"
    - If a theme is not present, mark it with a "0"
    - Do not code as present if the response references but rejects a theme. 
For example, within the Overcomes category, the response “if the other person refuses to listen or try understand why they're being insensitive, there is no point in putting myself under stress of trying to educate someone who does not want to be educated. i would honestly just walk away from the conversation” should not be coded as Education.

4. Justification: 
    - Your response should include an explanation of the text's scoring. 
    
5. Format: 
    - Use a bulleted list to show each theme with its corresponding score and justification. For example:   
        Text: "Example text from the dataset."
        Scoring:
        * Theme 1: 1
            * Justification: The text includes a concept that aligns with the definition of Theme 1.
        * Theme 2: 0
            * Justification: The text does not include any concepts related to Theme 2.
        

        
The themes you will be analyzing are:

(For “Benefits”):
Awareness of Racial Issues and Empathy
- Definition: Developing a deeper understanding or awareness of racial issues and/or greater level of empathy.
- Example: “I believe having these conversations is educational and broadens my understanding of the experiences that other races have that may be different from mine.”
- Example: “These conversations are important to the individual because they help broaden perspective and cultivate more empathy. Listening to someone you care about talk about something they care about helps the friendship become more open, as well as reveal perspective that one of the friends may not have considered or been exposed to before.”

Trust & Solidarity
- Definition: Trust or allyship between groups or individuals. 
- Example: “I think when you and a friend can openly talk about race in an understanding manner, a certain sense of trust and security reveals itself.”

(For “Obstacles”):
Discomfort
- Definition: Feeling uncomfortable, awkward, weird, or anxious. 
- Example: “Me feeling uncomfortable with admitting to some of the things that have happened to me or just feeling embarrassed I even experienced that.”

Bias/racism 
- Definition: Behaviors that are biased, racist, or disrespectful. 
- Example: “If they become biased or seem to be racist about it.”
- Example: “If people do not see eye to eye on things because the person, who is not of color, cannot empathize for the other friend” 

Communication Break-down 
- Definition: Conflicts, defensiveness, misunderstandings, an unwillingness to listen, or other negative communication patterns. 
- Example: “If there is previous tension that doesn't allow for a full in depth conversation.”
- Example: “refusal to listen”

Ignorance 
- Definition: A lack of understanding of racial issues.
- Example: “if they are not willing to understand where you are coming from, or not willing to stop being ignorant.”

Lack of Interest
- Definition: Disinterested, disengaged from the topic of conversation. 
- Example: “When you notice a friend seems uninterested, almost annoyed in what you're trying to speak about, it can make you feel like a burden and you might decide not to speak to that friend about those issues.”

No Obstacles 
- Definition: No perceived obstacles. 
- Example: “I don't really see anything that would get in our way of experiencing these benefits my friend is very open and understanding and even if he can't fully understand I know he will definitely try to.”


(For “Overcomes”):
Active Listening
- Definition: Interested, attentive, affirming when listening.
- Example: “Pay full attention, let your friends speak completely and be sensitive to these types of topics.”

Education
- Definition: Providing information or sharing knowledge, experiences, or personal anecdotes to increase understanding.
- Example: “I can explain to them why things are they way they are and educate them about the problems that I experience that they may not have.”

Communicating about Challenges 
- Definition: Sharing thoughts or feelings related to challenges.
- Example: “Be willing to fully explain the thoughts and feelings I have about race related issues and not necessarily expect my friends to fully understand what I'm thinking about but instead give them room to learn and empathize.”

Acceptance
- Definition: Non-confrontational approach to conflict resolution; acceptance of an issue without further attempt to resolve it.  
- Example: “Acknowledge that people believe what people believe and they probably won't change that.”

Avoidance
- Definition: Ending the friendship or avoiding the topic of conversation. 
- Example: “I will respect them, but move on to someone else, and I could also maybe talk to them to try and get them to see it in a different point of view.”

Text: {text}
Themes: {themes}



"""


# In[14]:


themes_benefits = ["Awareness of Racial Issues and Empathy", "Trust & Solidarity"]
themes_obstacles = ["Discomfort", "Bias/racism", "Communication Break-down", "Ignorance", "Lack of Interest", "No Obstacles"]
themes_overcomes = ["Active Listening", "Education", "Communicating about Challenges", "Acceptance", "Avoidance"]


# In[16]:


def classify(text_to_classify, theme_list):
    # dictionary where keys = theme, values = score
    # the classifcations dictionary will be the values for the classification column in our output dataframe
    classifications = {theme: 0 for theme in theme_list}  # baseline score each theme is 0

    # use the prompt created above, but modify it based on the row label (Benefits, Obstacles, Overcomes)
    prompt = prompt_template.format(text=text_to_classify, themes=', '.join(theme_list))
   
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "You are a text classification expert. Carefully analyze the text for each theme and mark '1' if a theme is present and '0' if it is not. Pay close attention to the examples provided for each theme."},

            {"role": "user", "content": prompt}
        ],
        temperature=0, # explicitly set temperature to 0
        max_tokens= 1500 # make sure we have enough tokens to span the entire length of the list of themes
    )

    
   # print("Prompt:\n", prompt)  # Debug: Print prompt
   # print("Response:\n", response.choices[0].message['content'])  # Debug: Print raw response
    
    result = response.choices[0].message['content'].strip().split('\n') # Extract response and split by line  
    
    for theme in theme_list:  # for each theme in the specified list
        for line in result:  # for each line of the chat.completion method
            if theme in line and '1' in line:  # if the theme is in the line and determined to be present
                classifications[theme] = 1  # score it as a 1 in classifications dictionary, then go to the next theme in the list
                break

    return classifications 



def apply_classification(row):
    if row['Label'] == 'Benefits':
        theme_list = ["Awareness of Racial Issues and Empathy", "Trust & Solidarity"]
    elif row['Label'] == 'Obstacles':
        theme_list =  ["Discomfort", "Bias/racism", "Communication Break-down", "Ignorance", "Lack of Interest", "No Obstacles"]
    elif row['Label'] == 'Overcomes':
        theme_list = ["Active Listening", "Education", "Communicating about Challenges", "Acceptance", "Avoidance"]
    
    return classify(row['Text'], theme_list) # based on the row label, we know what list of themes to pass to our classify function above



import streamlit as st


st.title("SID Lab: Thematic Analysis Tool")
st.header("Social Identity in Dialogue Lab | Professor Kiara Sanchez | Summer Term 2024")


uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    input = pd.read_csv(uploaded_file) 
    
    with st.spinner('Processing...'):
        output_df = input.apply(apply_classification, axis=1)
        st.success('Processing complete!')
        
    st.download_button(
        label="Download CSV",
        data=output_df.to_csv(index=False),
        file_name="thematic_analysis_output.csv",
        mime="text/csv"
    )


# In[ ]:


# classifications = vc_long.apply(apply_classification, axis=1) ## applies the helper function to each row


# ## create a column for each theme with their respective scoresw
# for theme in themes_benefits + themes_obstacles + themes_overcomes: # for each theme 
#                 # get the theme's score from the classifications dict-- if its not in the dictionary keep the value at 0
#     vc_long[theme] = classifications.apply(lambda x: x.get(theme, 0))


# In[ ]:





# In[ ]:





# In[ ]:




