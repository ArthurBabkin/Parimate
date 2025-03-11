import asyncio
from telegram import Update
from telegram.ext import (
    ContextTypes,
    Application,
    MessageHandler,
    CommandHandler
)
import random
from ollama import chat
import asyncio
from telegram.ext import filters, MessageHandler
import sqlite3
from src.face_analysis import *

BOT_TOKEN = "8034032425:AAG2u988cGUs-V3BBAnq8SWaxwGiIQCmRss"
conn = sqlite3.connect("users.db")


def convert_to_embeddings(image_byte64):
    # Convert image to embeddings
    embeddings = [random.random() for _ in range(512)]
    return ','.join(map(str, embeddings))

with conn:
    with open("init.sql", "r", encoding="utf-8") as file:
        conn.executescript(file.read())
    
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # send the type of last message from user. text, video, photo, etc.
    message_type = "unknown"
    if update.message.text:
        message_type = "text"
    elif update.message.photo:
        message_type = "photo"
    elif update.message.video:
        message_type = "video"
    elif update.message.audio:
        message_type = "audio"
    elif update.message.document:
        message_type = "document"
    elif update.message.voice:
        message_type = "voice"

    await update.message.reply_text(f"Last message type: {message_type}")
    
async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm Parimate")
    await update.message.reply_text("Now send photo of your face")
    
    photo_handler = MessageHandler(filters.ALL, handle_photo_request)
    context.application.add_handler(photo_handler)


async def handle_photo_request(update, context):
    if not update.message.photo:
        await update.message.reply_text("Send me a photo of your face 😀😀😀😀😀😀😀")
        return
    user_id = update.message.from_user.id
    file_id = update.message.photo[-1].file_id
    # Convert image to embeddings
    image_base64 = await context.application.bot.get_file(file_id)
    embeddings = extract_embedding(image_base64)
    try:
        conn = sqlite3.connect("users.db")
        with conn:
            # create table if not exists
            conn.execute("CREATE TABLE IF NOT EXISTS user_photos (id INTEGER, ufile_id TEXT, embedding VECTOR)")
            # insert user photo
            conn.execute("INSERT INTO user_photos (id, ufile_id) VALUES (?, ?)", (user_id, embeddings))
    except Exception as e:
        await update.message.reply_text("Error saving photo: " + str(e))
        return
    await update.message.reply_text("Photo saved!")
    
async def handle_create_task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Enter the task name:")
    task_name = update.message.text
    await update.message.reply_text(f"Task {task_name} created successfully!")
    
def main():
    
    application = Application.builder().token(BOT_TOKEN).build()
    # Register handlers
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(CommandHandler("create_task", handle_create_task))
    # application.add_handler(MessageHandler(None, handle_message))
    # Start polling
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    
asyncio.run(main())


