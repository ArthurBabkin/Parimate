import os
import random

from omegaconf import DictConfig
from telegram import Update
from telegram.ext import (ApplicationBuilder, CommandHandler, ContextTypes,
                          ConversationHandler, MessageHandler, filters, CallbackQueryHandler)

from internal.domain.service import ParimateSerive
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from deepface import DeepFace
import time
class ParimateBot:
    def __init__(self, cfg: DictConfig, service: ParimateSerive):
        self.app = ApplicationBuilder().token(cfg.token).build()
        self.service = service
        self.is_done = False
        self.anecdotes = [
            '''Диалоги на форуме:
- Хотел скачать "Одиссею" Гомера. Книга удалена по требованию правообладателей. Кто, блядь, правообладатель?! Гомер?!
- Чувствую, скоро захочу скачать библию, но Иисус заберёт права и заставит покупать её через Литрес.
- Из-за вот таких, как вы, бесплатно скачивающих, Гомер больше ничего и не напишет, вот!''',
            '''Я в возрасте 14 лет, читая Толкина: Хоббиты считаются детьми до 33 лет? Охренеть, что за странный мир, вообще не похож на наш!
Я в возрасте 25 лет: а нет, всё нормально.
Я в 34 года: кажется, хоббиты слишком торопятся стать взрослыми.''',
            '''Жена мужу:
- Ты совсем не уделяешь мне внимания! Подозреваю, ты завел любовницу.
Муж:
- Но, дорогая, как только я обращаю на тебя внимание, у тебя сразу начинает болеть голова. Я забочусь о твоем здоровье!''',
            '''
Продавец лотерейных билетов:
- Мужчина, купите лотерыйный билет и всего за 100 рублей вы сможете выиграть отличный автомобиль стоимостью 1 миллион рублей.
- Да не нужен мне автомобиль! Он будет сжирать кучу денег: бензин, стpaховка, запчасти, ремонт, штрафы, паркинг и т.д.
- Послушайте, мужчина, 500 тысяч билетов, а выигрыш только один. Поэтому у вас практически нет шансов получить эту машину, а, следовательно, риск понести все эти расходы близок к нулю. Так вы берёте билет?'''
        ]
        self.update = None

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Hello! I'm Parimate")
        await update.message.reply_text("Now send a photo of your face")

        async def handle_photo_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not update.message.photo:
                await update.message.reply_text("Send me a photo of your face 😀")
                return

            user_id = update.message.from_user.id
            file_id = update.message.photo[-1].file_id

            image_base64 = await context.bot.get_file(file_id)
            try:
                embeddings = self.convert_to_embeddings(image_base64)
                self.service.insert_photo(user_id, embeddings)
            except Exception as e:
                await update.message.reply_text("Error saving photo: " + str(e))

            await update.message.reply_text("Photo saved!")
            context.application.remove_handler(photo_handler)

        photo_handler = MessageHandler(filters.ALL, handle_photo_request)
        context.application.add_handler(photo_handler)

    async def handle_create_task(self, update: Update,
                                 context: ContextTypes.DEFAULT_TYPE):
        async def handle_task_name(update: Update,
                                   context: ContextTypes.DEFAULT_TYPE):
            context.user_data["task_name"] = update.message.text
            await update.message.reply_text("Enter the task description:")
            context.application.remove_handler(task_name_handler)
            context.application.add_handler(task_description_handler)

        async def handle_task_description(update: Update,
                                          context: ContextTypes.DEFAULT_TYPE):
            context.user_data["task_description"] = update.message.text
            await update.message.reply_text("Enter the code phrase:")
            context.application.remove_handler(task_description_handler)
            context.application.add_handler(task_code_phrase_handler)

        async def handle_task_code_phrase(update: Update,
                                          context: ContextTypes.DEFAULT_TYPE):
            context.user_data["task_code_phrase"] = update.message.text
            user_id = update.message.from_user.id
            try:
                self.service.create_task(
                    user_id,
                    context.user_data["task_name"],
                    context.user_data["task_description"],
                    context.user_data["task_code_phrase"],
                )
            except Exception as e:
                await update.message.reply_text("Error creating task: " + str(e))
                return
            await update.message.reply_text("Task created!")
            context.application.remove_handler(task_code_phrase_handler)

        task_name_handler = MessageHandler(filters.TEXT, handle_task_name)
        task_description_handler = MessageHandler(filters.TEXT,
                                                  handle_task_description)
        task_code_phrase_handler = MessageHandler(filters.TEXT,
                                                  handle_task_code_phrase)

        await update.message.reply_text("Enter the task name:")
        context.application.add_handler(task_name_handler)

    async def handle_done_task(self, update: Update,
                               context: ContextTypes.DEFAULT_TYPE):
        global current_task
        current_task = None
        async def handle_task_selection(update: Update,
                                        context: ContextTypes.DEFAULT_TYPE):
            global current_task
            await update.callback_query.message.delete()
            task_index = int(update.callback_query.data.split("_")[1])
            context.user_data["task_index"] = task_index
            current_task = self.service.get_tasks(update.callback_query.from_user.id)[task_index]
            await update.callback_query.message.reply_text(f"Отправь видео, которое подтвердит \"{current_task["name"]}\"")
               
            context.application.remove_handler(callback_handler)
            
            context.application.add_handler(video_handler)
            
        async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
            
            global current_task
            
            if not update.message.video:
                await update.message.reply_text("Это не видео. "
                                "Пожалуйста, отправьте видео.")
                return 

            user_id = update.message.from_user.id
            file_id = update.message.video.file_id

            try:
                video_file = await context.bot.get_file(file_id)
                video_path = await video_file.download_to_drive()
                
                await update.message.reply_text(f"Обрабатываем видео..")
                
                self.service.done_task(user_id, current_task["name"],  video_path, self.on_done)
                self.update = update
                for i in range(1, 100):
                    if(self.is_done):
                        break
                    await update.message.reply_text(f"Пока видео обрабатывается, я расскажу вам анекдот: {random.choice(self.anecdotes)}")
                    time.sleep(30)
                    
            except Exception as e:
                print(f"Ошибка при обработке видео: {e}")
                await update.message.reply_text(
                    "Произошла ошибка при обработке видео. "
                    "Пожалуйста, попробуйте снова."
                )

            return ConversationHandler.END

        await update.message.reply_text("Конечно! "
                                        "Выберите задачу:")
        user_id = update.message.from_user.id
        tasks = self.service.get_tasks(user_id)
        if not tasks:
            await update.message.reply_text("No tasks found.")
            return ConversationHandler.END

        keyboard = [
            [InlineKeyboardButton(task['name'], callback_data=f"task_{i}")]
            for i, task in enumerate(tasks)
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        callback_handler = CallbackQueryHandler(handle_task_selection, pattern=r"^task_\d+$")
        video_handler = MessageHandler(filters.VIDEO, handle_video)

       
        context.application.add_handler(callback_handler)

        await update.message.reply_text("Select task:", reply_markup=reply_markup)
        
    def on_done(self, result, update: Update):
        self.is_done = True
        if result:
            self.update.message.reply_text("Задача выполнена успешно!")
        else:
            self.update.message.reply_text("Задача не выполнена!")
        
        

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Действие отменено.")
        return ConversationHandler.END

    def register_handlers(self):
        
        self.app.add_handler(CommandHandler("start",
                                            self.handle_start))
        self.app.add_handler(CommandHandler("create_task",
                                            self.handle_create_task))
        self.app.add_handler(CommandHandler("done_task",
                                            self.handle_done_task))

    def run(self):
        self.register_handlers()
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)

    def convert_to_embeddings(self,image_byte64):
        # Convert image to embeddings
        return self.service.get_embedings(image_byte64)