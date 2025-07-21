import streamlit as st
import joblib
import numpy as np

modelo = joblib.load('modelo_titulo_capitalizacao.pkl')
scaler = joblib.load('scaler_titulo_capitalizacao.pkl')

st.title("Previsão de Compra de Título de Capitalização")

idade = st.slider("Idade", 18, 70, 30)
renda = st.number_input("Renda mensal", 1000, 30000, 5000)
cartao = st.checkbox("Tem cartão de crédito?")
usa_app = st.checkbox("Usa o app do banco?")
saldo = st.number_input("Saldo médio", 0, 50000, 3000)

if st.button("Prever"):
    dados = np.array([
        idade,
        renda,
        int(cartao),
        int(usa_app),
        saldo
    ]).reshape(1, -1)

    dados_escalados = scaler.transform(dados)
    resultado = modelo.predict(dados_escalados)[0]
    prob = modelo.predict_proba(dados_escalados)[0][1]

    st.write(f"### Resultado: {'Vai comprar' if resultado == 1 else 'Não vai comprar'}")
    st.write(f"Probabilidade: {round(prob * 100, 2)}%")
