import streamlit as st
from tools import obter_noticia_link, predict_fake_news, treinar_modelo

st.set_page_config(page_title="Detector de Fake News", layout="centered")
st.title("📰 CheckNews")
st.write("Informe um link de notícia para verificar sua autenticidade.")

link_noticia = st.text_input("Link da notícia:")

if st.button("Analisar"):
    with st.spinner("Analisando a notícia..."):
        model, vectorizer = treinar_modelo()
        noticia = obter_noticia_link(link_noticia)

        if noticia:
            resultado = predict_fake_news(noticia, model, vectorizer)
            st.success(f"A notícia fornecida é: **{resultado}**")
        else:
            st.error("Não foi possível obter o conteúdo da notícia.")
    
    
