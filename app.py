# Импорт необходимых библиотек
import streamlit as st  # Для создания веб-приложения
import requests  # Для отправки HTTP-запросов
from PIL import Image  # Для работы с изображениями
import io  # Для работы с потоками ввода/вывода
from streamlit_drawable_canvas import st_canvas  # Для создания интерактивного холста

# Заголовок приложения
st.title("Классификация изображений")  # Отображение заголовка приложения

# Опция для выбора режима ввода изображения
mode = st.radio("Выберите способ ввода изображения:", ("Нарисовать изображение", "Загрузить изображение"))

# Функция для отправки изображения на сервер и получения предсказания
def get_prediction(image_data):
    """
    Отправляет изображение на сервер FastAPI и возвращает предсказание.
    """
    try:
        # Отправка POST-запроса с изображением
        response = requests.post(
            "https://image-class-9vgx.onrender.com/predict/",
            files={"file": ("image.png", image_data, "image/png")}
        )
        response.raise_for_status()  # Проверка на наличие ошибок HTTP
        return response.json()  # Возврат предсказания из ответа
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при отправке запроса: {e}")  # Обработка ошибок запроса
    except ValueError as e:
        st.error(f"Ошибка при обработке ответа: {e}")  # Обработка ошибок ответа
    return None

# Обработка режима "Загрузить изображение"
if mode == "Загрузить изображение":
    # Загрузка изображения через интерфейс
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Создание колонок для отображения изображения и предсказания
        col1, col2 = st.columns(2)

        with col1:
            # Отображение загруженного изображения
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_container_width=True)  # Используем use_container_width вместо use_column_width

        # Преобразование изображения в байтовый поток
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Получение предсказания
        prediction = get_prediction(img_byte_arr)

        if prediction:
            with col2:
                # Отображение предсказания с увеличенным шрифтом
                st.markdown(
                    f"<div style='display: flex;   flex-direction: column; flex-wrap: wrap;height: 500px;'>"
                    f"<h1 style='font-size: 48px;'>Ваш класс - {prediction['predicted_class']}</h1>"
                    f"<h2 style='font-size: 32px;'>Кот - {round(prediction['probabilities'][0] * 100)}%</h2>"
                    f"<h2 style='font-size: 32px;'>Собака - {round(prediction['probabilities'][1] * 100)}%</h2>"
                    f"<h2 style='font-size: 32px;'>Панда - {round(prediction['probabilities'][2] * 100)}%</h2>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# Обработка режима "Нарисовать изображение"
elif mode == "Нарисовать изображение":
    # Настройки для холста
    stroke_width = st.slider("Толщина линии:", 1, 25, 9)

    # Размещение выбора цвета линии и фона
    col_color1, col_color2 = st.columns(2)
    with col_color1:
        stroke_color = st.color_picker("Цвет линии:", "#FFFFFF")
    with col_color2:
        bg_color = st.color_picker("Цвет фона:", "#000000")

    realtime_update = st.checkbox("Обновлять в реальном времени", True)

    # Инициализация состояния для управления холстом
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'history_index' not in st.session_state:
        st.session_state.history_index = -1

    col1, col2 = st.columns(2)

    with col1:
        # Холст для рисования
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

        # Кнопки управления холстом
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button("Стереть всё"):
                st.session_state.canvas_key += 1
                st.session_state.history = []
                st.session_state.history_index = -1
        with col_btn2:
            if st.button("← Назад") and st.session_state.history_index > 0:
                st.session_state.history_index -= 1
                st.session_state.canvas_key += 1
        with col_btn3:
            if st.button("Вперед →") and st.session_state.history_index < len(st.session_state.history) - 1:
                st.session_state.history_index += 1
                st.session_state.canvas_key += 1

        # Кнопка классификации при отключенном реальном времени
        if not realtime_update:
            if st.button("Классифицировать"):
                st.session_state.force_predict = True

    # Логика обработки изображения
    if canvas_result.image_data is not None:
        if realtime_update or st.session_state.get('force_predict', False):
            # Сохранение текущего состояния в историю
            current_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            if len(st.session_state.history) == 0 or current_image != st.session_state.history[-1]:
                st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
                st.session_state.history.append(current_image)
                st.session_state.history_index = len(st.session_state.history) - 1

            # Преобразование изображения и получение предсказания
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')

            prediction = get_prediction(img_byte_arr.getvalue())

            if prediction:
                with col2:
                    st.markdown(
                        f"<div style='display: flex; flex-direction: column; height: 280px;'>"
                        f"<h1 style='font-size: 36px;'>Результат: {prediction['predicted_class']}</h1>"
                        f"<p style='font-size: 24px;'>🐱 Кот: {round(prediction['probabilities'][0] * 100)}%</p>"
                        f"<p style='font-size: 24px;'>🐶 Собака: {round(prediction['probabilities'][1] * 100)}%</p>"
                        f"<p style='font-size: 24px;'>🐼 Панда: {round(prediction['probabilities'][2] * 100)}%</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            if 'force_predict' in st.session_state:
                del st.session_state.force_predict