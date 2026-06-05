# Импорт необходимых библиотек
import streamlit as st
import requests
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas

st.title("Классификация изображений")

mode = st.radio("Выберите способ ввода изображения:", ("Нарисовать изображение", "Загрузить изображение"))

def get_prediction(image_data):
    """
    Отправляет изображение на сервер FastAPI и возвращает предсказание.
    """
    try:
        # Указываем актуальный адрес Hugging Face (без слэша в конце)
        response = requests.post(
            "https://cashka-classification-backend.hf.space/predict",
            files={"file": ("image.png", image_data, "image/png")}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при отправке запроса: {e}")
    except ValueError as e:
        st.error(f"Ошибка при обработке ответа: {e}")
    return None

if mode == "Загрузить изображение":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_container_width=True)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        prediction = get_prediction(img_byte_arr)

        if prediction:
            with col2:
                # Внедрен нативный стек шрифтов для лучшего отображения
                st.markdown(
                    f"<div style='display: flex; flex-direction: column; height: 280px; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif;'>"
                    f"<h1 style='font-size: 36px;'>Результат: {prediction['predicted_class']}</h1>"
                    f"<p style='font-size: 24px;'>🐱 Кот: {round(prediction['probabilities'][0] * 100)}%</p>"
                    f"<p style='font-size: 24px;'>🐶 Собака: {round(prediction['probabilities'][1] * 100)}%</p>"
                    f"<p style='font-size: 24px;'>🐼 Панда: {round(prediction['probabilities'][2] * 100)}%</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

elif mode == "Нарисовать изображение":
    stroke_width = st.slider("Толщина линии:", 1, 25, 9)

    col_color1, col_color2 = st.columns(2)
    with col_color1:
        stroke_color = st.color_picker("Цвет линии:", "#FFFFFF")
    with col_color2:
        bg_color = st.color_picker("Цвет фона:", "#000000")

    realtime_update = st.checkbox("Обновлять в реальном времени", True)

    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'history_index' not in st.session_state:
        st.session_state.history_index = -1

    col1, col2 = st.columns(2)

    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button("Стереть всё"):
                st.session_state.canvas_key += 1
                st.session_state.history = []
                st.session_state.history_index = -1

        if not realtime_update:
            if st.button("Классифицировать"):
                st.session_state.force_predict = True

    if canvas_result.image_data is not None:
        if realtime_update or st.session_state.get('force_predict', False):
            current_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            if len(st.session_state.history) == 0 or current_image != st.session_state.history[-1]:
                st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
                st.session_state.history.append(current_image)
                st.session_state.history_index = len(st.session_state.history) - 1

            # ИСПРАВЛЕНИЕ: Сначала загружаем как RGBA, затем конвертируем в RGB
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')

            prediction = get_prediction(img_byte_arr.getvalue())

            if prediction:
                with col2:
                    st.markdown(
                        f"<div style='display: flex; flex-direction: column; height: 280px; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif;'>"
                        f"<h1 style='font-size: 36px;'>Результат: {prediction['predicted_class']}</h1>"
                        f"<p style='font-size: 24px;'>🐱 Кот: {round(prediction['probabilities'][0] * 100)}%</p>"
                        f"<p style='font-size: 24px;'>🐶 Собака: {round(prediction['probabilities'][1] * 100)}%</p>"
                        f"<p style='font-size: 24px;'>🐼 Панда: {round(prediction['probabilities'][2] * 100)}%</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            if 'force_predict' in st.session_state:
                del st.session_state.force_predict
