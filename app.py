import streamlit as st
import os
import uuid
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import json
from datetime import datetime
from utils import generate_pdf_report, save_history, load_history, clear_history

# Проверка устройства
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Создание папок
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("history", exist_ok=True)

# Загрузка модели
model = YOLO("models/yolov8n.pt")

# Название приложения
st.set_page_config(page_title="🐻 Bear Detection System", layout="wide")
st.title("🐻 Bear Detection System")
st.sidebar.info(f"Текущее устройство: {DEVICE.upper()}")

# Выбор режима
page = st.sidebar.selectbox("Выберите режим:", ("Изображение", "Видео", "Камера", "История"))

def detect_and_display(image_np):
    results = model.predict(image_np, save=False, device=DEVICE)
    img_result = np.array(image_np.copy())

    bear_count = 0
    logs = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if cls == 21 and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Bear {conf:.2f}"
                cv2.putText(img_result, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                bear_count += 1
        logs.append(f"{r.speed['preprocess']:.1f}ms preprocess, {r.speed['inference']:.1f}ms inference, {r.speed['postprocess']:.1f}ms postprocess")

    return img_result, bear_count, logs

# ---------- Режимы ----------

# 1. Изображение
if page == "Изображение":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        if st.button("🚀 Обработать"):
            img_np = np.array(image)
            img_result, bear_count, logs = detect_and_display(img_np)

            result_path = os.path.join("data", f"{uuid.uuid4()}_processed.jpg")
            Image.fromarray(img_result).save(result_path)

            st.image(img_result, caption=f"Найдено медведей: {bear_count}", use_container_width=True)
            st.success(f"Медведей найдено: {bear_count}")

            with open(result_path, "rb") as f:
                st.download_button("📥 Скачать обработанное изображение", f, file_name="processed_image.jpg", mime="image/jpeg")

            save_history({
                "type": "image",
                "filename": uploaded_file.name,
                "original_path": os.path.join("data", uploaded_file.name),
                "processed_path": result_path,
                "bear_count": bear_count,
                "timestamp": datetime.now().isoformat(),
                "detection_logs": logs
            })

# 2. Видео
elif page == "Видео":
    uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video:
        video_id = str(uuid.uuid4())
        input_path = os.path.join("data", f"{video_id}_input.mp4")
        output_path = os.path.join("data", f"{video_id}_processed.mp4")

        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(input_path)

        if st.button("🚀 Обработать видео"):
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            bear_counter_placeholder = st.empty()
            stframe = st.empty()

            total_bears = 0
            bears_per_frame = []
            detection_logs = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                img_result, bear_count, logs = detect_and_display(frame)
                out.write(img_result)

                total_bears += bear_count
                bears_per_frame.append(bear_count)
                for log in logs:
                    detection_logs.append(f"Кадр {frame_idx}: {log}")

                frame_idx += 1
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
                bear_counter_placeholder.metric("🐻 Найдено медведей", total_bears)
                stframe.image(img_result, channels="BGR", use_container_width=True)

            cap.release()
            out.release()

            st.success(f"Видео обработано! Найдено медведей: {total_bears}")

            st.video(output_path)
            st.markdown(f"[➡️ Открыть обработанное видео для перемотки]({output_path})")

            with open(output_path, "rb") as f:
                st.download_button("📥 Скачать обработанное видео", f, file_name="processed_video.mp4", mime="video/mp4")

            save_history({
                "type": "video",
                "filename": uploaded_video.name,
                "original_path": input_path,
                "processed_path": output_path,
                "bear_count": total_bears,
                "timestamp": datetime.now().isoformat(),
                "bear_counts_per_frame": bears_per_frame,
                "detection_logs": detection_logs
            })

# 3. Камера
elif page == "Камера":
    st.warning("Поток с камеры через Streamlit ограничен. Лучше использовать отдельное локальное приложение.")

# 4. История
elif page == "История":
    st.header("📋 История обработок")

    history = load_history()

    if history:
        with st.expander("⚙️ Настройки отчета"):
            include_images = st.checkbox("Добавить изображения", value=True)
            detailed_report = st.checkbox("Подробный отчет", value=True)

            if st.button("📄 Сгенерировать PDF-отчет"):
                report_path = "history/history_report.pdf"
                generate_pdf_report(history, include_images, detailed_report, output_path=report_path)
                with open(report_path, "rb") as f:
                    st.download_button("📥 Скачать отчет в PDF", f, file_name="history_report.pdf", mime="application/pdf")

        for idx, record in enumerate(reversed(history)):
            with st.expander(f"{record['type'].capitalize()}: {record['filename']}"):
                st.text(f"Файл: {record['filename']}")
                st.text(f"Медведей найдено: {record['bear_count']}")
                st.text(f"Дата и время: {record['timestamp']}")

                if record["type"] == "image":
                    col1, col2 = st.columns(2)
                    with col1:
                        if os.path.exists(record["original_path"]):
                            st.image(record["original_path"], use_container_width=True)
                        else:
                            st.error("Оригинальный файл отсутствует.")

                    with col2:
                        if os.path.exists(record["processed_path"]):
                            st.image(record["processed_path"], use_container_width=True)
                        else:
                            st.error("Файл с обработанным изображением отсутствует.")

                elif record["type"] == "video":
                    if os.path.exists(record["original_path"]):
                        st.video(record["original_path"])
                    else:
                        st.error("Оригинальное видео отсутствует.")

                    if os.path.exists(record["processed_path"]):
                        st.video(record["processed_path"])
                    else:
                        st.error("Файл с обработанным видео отсутствует.")

        if st.button("🗑️ Очистить всю историю"):
            clear_history()
            st.success("История очищена!")
    else:
        st.info("История пока пуста.")
