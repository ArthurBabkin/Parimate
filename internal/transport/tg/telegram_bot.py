import random

from omegaconf import DictConfig
from telegram import Update
from telegram.ext import (ApplicationBuilder, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from internal.domain.service.service import ParimateSerive


class ParimateBot:
    def __init__(self, cfg: DictConfig, service: ParimateSerive):
        self.app = ApplicationBuilder().token(cfg.tg.token).build()
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
            await update.message.reply_text(
                "Send me a photo of your face 😀😀😀😀😀😀😀")
            return
        user_id = update.message.from_user.id
        file_id = update.message.photo[-1].file_id
        # Convert image to embeddings
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

    def register_handlers(self):
        self.app.add_handler(CommandHandler("start", self.handle_start))
        self.app.add_handler(
            CommandHandler("create_task", self.handle_create_task))

    def run(self):
        self.register_handlers()
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

    @staticmethod
    def convert_to_embeddings(image_byte64):
        # Convert image to embeddings
        embeddings = [random.random() for _ in range(512)]
        return ','.join(map(str, embeddings))
