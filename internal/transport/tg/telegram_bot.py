import os
import random

from omegaconf import DictConfig
from telegram import Update
from telegram.ext import (ApplicationBuilder, CommandHandler, ContextTypes,
                          MessageHandler, filters, ConversationHandler)

from internal.domain.service import ParimateSerive

WAITING_FOR_VIDEO = 1


class ParimateBot:
    def __init__(self, cfg: DictConfig, service: ParimateSerive):
        self.app = ApplicationBuilder().token(cfg.token).build()
        self.service = service

    async def handle_start(self, update: Update,
                           context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Hello! I'm Parimate")
        await update.message.reply_text("Now send photo of your face")

        photo_handler = MessageHandler(filters.ALL, self.handle_photo_request)
        context.application.add_handler(photo_handler)

    async def handle_message(self, update: Update,
                             context: ContextTypes.DEFAULT_TYPE):
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

    async def handle_photo_request(self, update, context):
        if not update.message.photo:
            await update.message.reply_text("Send me a photo of your face 😀")
            return
        user_id = update.message.from_user.id
        file_id = update.message.photo[-1].file_id

        image_base64 = await context.application.bot.get_file(file_id)
        embeddings = self.convert_to_embeddings(image_base64)
        try:
            self.service.insert_photo(user_id, embeddings)
        except Exception as e:
            await update.message.reply_text("Error saving photo: " + str(e))
            return
        await update.message.reply_text("Photo saved!")

    async def handle_create_task(self, update: Update,
                                 context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Enter the task name:")
        task_name = update.message.text
        await update.message.reply_text(
            f"Task {task_name} created successfully!")

    async def handle_done_task(self, update: Update,
                               context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Пожалуйста, отправьте видео "
                                        "для обработки.")
        return WAITING_FOR_VIDEO

    async def handle_video(self, update: Update,
                           context: ContextTypes.DEFAULT_TYPE):
        if not update.message.video:
            await update.message.reply_text("Это не видео. "
                                            "Пожалуйста, отправьте видео.")
            return WAITING_FOR_VIDEO

        user_id = update.message.from_user.id
        file_id = update.message.video.file_id

        try:
            video_file = await context.bot.get_file(file_id)
            video_path = await video_file.download_to_drive()

            result = self.service.done_task(user_id, video_path)

            result = f"Видео успешно обработано! Video result: {result}"

            await update.message.reply_text(result)

        except Exception as e:
            print(f"Ошибка при обработке видео: {e}")
            await update.message.reply_text("Произошла ошибка при обработке "
                                            "видео. Пожалуйста, попробуйте "
                                            "снова.")

        return ConversationHandler.END  # Завершаем диалог

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Действие отменено.")
        return ConversationHandler.END

    def register_handlers(self):
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler("done_task", self.handle_done_task)],
            states={
                WAITING_FOR_VIDEO: [
                    MessageHandler(filters.VIDEO, self.handle_video)],
            },
            fallbacks=[CommandHandler("cancel", self.cancel)],
        )
        self.app.add_handler(CommandHandler("start",
                                            self.handle_start))
        self.app.add_handler(conv_handler)
        self.app.add_handler(CommandHandler("create_task",
                                            self.handle_create_task))

    def run(self):
        self.register_handlers()
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

    @staticmethod
    def convert_to_embeddings(image_byte64):
        # Convert image to embeddings
        embeddings = [random.random() for _ in range(512)]
        return ','.join(map(str, embeddings))
