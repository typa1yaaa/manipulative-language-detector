import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Manipulation Detector", layout="centered")
st.title("Детектор манипулятивных паттернов")
st.caption("Вставь текст письма/сообщения — модель подсветит манипулятивные фрагменты")

text = st.text_area("Текст для анализа", height=200, placeholder="Вставьте сообщение сюда...")

if st.button("Проверить", type="primary", disabled=not text.strip()):
    with st.spinner("Анализирую..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка запроса к API: {e}")
            st.stop()

    spans = data["spans"]

    if not data["has_manipulation"]:
        st.success("Манипулятивных паттернов не обнаружено")
    else:
        st.warning(f"Найдено паттернов: {len(spans)}")

        html = text
        offset = 0
        for span in spans:
            start = text.find(span["text"])
            if start == -1:
                continue
            end = start + len(span["text"])
            start, end = start + offset, end + offset
            tag = f'<mark title="{span["label_ru"]} ({span["confidence"]*100:.0f}%)" style="background:#ffd6d6">'
            html = html[:start] + tag + html[start:end] + "</mark>" + html[end:]
            offset += len(tag) + len("</mark>")

        st.markdown(html, unsafe_allow_html=True)

        st.divider()
        for span in spans:
            st.write(f"**{span['label_ru']}** ({span['confidence']*100:.0f}%): _{span['text']}_")
