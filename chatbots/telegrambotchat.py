import logging
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("I can help you with anything!")
    

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)


############################################################################

async def bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    user_input = update.message.text

        # Prepare the data to be sent in the request
    data = {"system": "THe context is a conversation ","text": user_input }
    if update.message.text != '':
            response = requests.post("http://localhost:5000/telebotchat", json=data)
            answer = response.text
    else:
        return
        
    await update.message.reply_text(str(answer))



def main() -> None:
    """Start the bot."""
    # Write your TELEGRAM token here ---->
    application = Application.builder().token("").build()

    # on different commands - answer in Telegram
    #application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    # application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_reply))
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()