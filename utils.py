import os
import json
import random
from fpdf import FPDF
from PIL import Image
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Пути к шрифтам
FONT_REGULAR = "Montserrat.ttf"
FONT_BOLD = "Montserrat-Bold.ttf"

# --- Работа с историей ---

def save_history(entry):
    os.makedirs("history", exist_ok=True)
    path = os.path.join("history", "history.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    history.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

def load_history():
    path = os.path.join("history", "history.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

def clear_history():
    path = os.path.join("history", "history.json")
    if os.path.exists(path):
        os.remove(path)

# --- Генерация красивого PDF-отчета ---

def generate_pdf_report(
    history_data,
    include_images=True,
    detailed=True,
    output_path="history/history_report.pdf",
    max_logs_per_record=300
):
    os.makedirs("history", exist_ok=True)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Шрифты
    pdf.add_font('Montserrat', '', FONT_REGULAR, uni=True)
    pdf.add_font('Montserrat', 'B', FONT_BOLD, uni=True)
    pdf.set_font('Montserrat', 'B', 16)
    pdf.cell(0, 10, "Отчет по обработке изображений и видео", ln=True, align="C")
    pdf.ln(10)

    for idx, record in enumerate(history_data, 1):
        # Заголовок записи
        pdf.set_font('Montserrat', 'B', 14)
        pdf.cell(0, 8, f"{idx}. Файл: {record['filename']}", ln=True)
        pdf.set_font('Montserrat', '', 12)
        pdf.cell(0, 6, f"Тип: {record['type'].capitalize()}", ln=True)
        pdf.cell(0, 6, f"Медведей найдено: {record['bear_count']}", ln=True)
        pdf.cell(0, 6, f"Дата и время: {record['timestamp']}", ln=True)
        pdf.ln(5)

        # Изображения
        if include_images:
            try:
                if record["type"] == "image":
                    if os.path.exists(record["original_path"]):
                        pdf.set_font('Montserrat', 'B', 12)
                        pdf.cell(0, 8, "Оригинал:", ln=True)
                        pdf.image(record["original_path"], w=120)
                        pdf.ln(5)
                    if os.path.exists(record["processed_path"]):
                        pdf.set_font('Montserrat', 'B', 12)
                        pdf.cell(0, 8, "Обработанное изображение:", ln=True)
                        pdf.image(record["processed_path"], w=120)
                        pdf.ln(10)
                elif record["type"] == "video":
                    clip = VideoFileClip(record["original_path"])
                    frame_time = random.uniform(0, clip.duration)
                    frame = clip.get_frame(frame_time)
                    temp_orig = f"data/temp_orig_{idx}.jpg"
                    Image.fromarray(frame).save(temp_orig)

                    clip2 = VideoFileClip(record["processed_path"])
                    frame2 = clip2.get_frame(frame_time)
                    temp_proc = f"data/temp_proc_{idx}.jpg"
                    Image.fromarray(frame2).save(temp_proc)

                    pdf.set_font('Montserrat', 'B', 12)
                    pdf.cell(0, 8, "Кадр из оригинала:", ln=True)
                    pdf.image(temp_orig, w=120)
                    pdf.ln(5)

                    pdf.cell(0, 8, "Кадр из обработки:", ln=True)
                    pdf.image(temp_proc, w=120)
                    pdf.ln(10)

                    os.remove(temp_orig)
                    os.remove(temp_proc)

            except Exception as e:
                pdf.set_font('Montserrat', '', 10)
                pdf.cell(0, 6, f"⚠️ Ошибка предпросмотра видео: {str(e)}", ln=True)

        # График
        if detailed and "bear_counts_per_frame" in record:
            try:
                plt.figure(figsize=(6, 3))
                plt.plot(range(len(record["bear_counts_per_frame"])), record["bear_counts_per_frame"], marker="o")
                plt.title("Количество медведей по кадрам")
                plt.xlabel("Кадр")
                plt.ylabel("Количество")
                plt.grid(True)
                temp_plot = f"data/temp_plot_{idx}.png"
                plt.savefig(temp_plot)
                plt.close()
                pdf.set_font('Montserrat', 'B', 12)
                pdf.cell(0, 8, "График медведей по кадрам:", ln=True)
                pdf.image(temp_plot, w=180)
                pdf.ln(10)
                os.remove(temp_plot)
            except Exception as e:
                pdf.cell(0, 6, f"⚠️ Ошибка построения графика: {str(e)}", ln=True)

        # Логи
        if detailed and "detection_logs" in record:
            pdf.set_font('Montserrat', 'B', 12)
            pdf.cell(0, 8, "Логи обработки:", ln=True)
            pdf.set_font('Montserrat', '', 10)

            logs = record["detection_logs"][:max_logs_per_record]

            for log in logs:
                safe_log = log if len(log) < 200 else log[:200] + "..."
                if pdf.get_y() + 10 > pdf.page_break_trigger:
                    pdf.add_page()
                    pdf.set_font('Montserrat', 'B', 12)
                    pdf.cell(0, 8, "Продолжение логов:", ln=True)
                    pdf.set_font('Montserrat', '', 10)
                pdf.cell(0, 6, safe_log, ln=True)

            pdf.ln(5)

        # Разделение записей
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)

    pdf.output(output_path)
