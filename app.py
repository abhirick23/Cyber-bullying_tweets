import streamlit as st
import altair as alt

import pandas as pd
import numpy as np

import joblib

pipe_lr = joblib.load(open("cyber_text_classiication_pipe_lr.pkl","rb"))

#prdict functon
def predict_Cyber_text(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_probability_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results




#main function
def main():
    st.title("Cyber_text Detection App")
    menu = ["Home","Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Cyber_text in TEXT")

        with st.form(key='form_Cyber_text_clf'):
            raw_text = st.text_area("Type the text here ")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2) 

            prediction = predict_Cyber_text(raw_text)
            probability= get_probability_proba(raw_text)


            with col1:
                st.success("original text")
                st.write(raw_text)

                st.success("Prediction")
                st.write(prediction)
                st.write("confidence:{}".format(np.max(probability)*100))
            
            with col2:
                st.success("Prdiction probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                #st.write(proba_df.T)
                proba_df_clean= proba_df.T.reset_index()
                proba_df_clean.columns=['Cyber_texts','probability']

                fig=alt.Chart(proba_df_clean).mark_bar().encode(x='Cyber_texts',y='probability',color = 'Cyber_texts')
                st.altair_chart(fig,use_container_width=True)





    elif choice == "Monitor":
        st.subheader("Monitor app")

    else:
        st.subheader("About")

if __name__ == '__main__':
    main()