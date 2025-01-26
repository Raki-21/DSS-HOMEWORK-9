import torch
from transformers import pipeline
from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Define constants
TOKEN: Final = '7898009093:AAGdHPJZ6cj_LpE9UYCpuWkg2BxOTSi89DE'
MODEL_NAME: Final = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def initialize_pipeline():
    """Initialize and return the text-generation pipeline."""
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

# Initialize pipeline globally
pipe = initialize_pipeline()

def generate_response(user_input: str) -> str:
    """Generate a response from the LLM based on user input."""
    print(f"Received user input: {user_input}")

    messages = [{"role": "user", "content": user_input}]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipe(
        prompt, max_new_tokens=100, do_sample=True,
        temperature=0.8, top_k=50, top_p=0.95
    )

    raw_response = outputs[0]['generated_text'].split('<|assistant|>')[-1].strip()

    # Ensure the response ends with appropriate punctuation
    if not raw_response.endswith(('.', '!', '?')):
        last_punct = max(raw_response.rfind('.'), raw_response.rfind('!'), raw_response.rfind('?'))
        raw_response = raw_response[:last_punct + 1] if last_punct != -1 else raw_response + '.'

    print(f"Generated response: {raw_response}")
    return raw_response

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    welcome_message = "Hello! Welcome to Achutch Bot. How can I assist you today?"
    await update.message.reply_text(welcome_message)

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming user messages."""
    user_message = update.message.text
    bot_response = generate_response(user_message)
    await update.message.reply_text(bot_response)

async def log_error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors occurring during updates."""
    print(f"Error with update {update}: {context.error}")

def run_bot():
    """Set up and start the bot."""
    print("Initializing bot...")

    # Create the application instance
    application = Application.builder().token(TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))

    # Add error handler
    application.add_error_handler(log_error)

    print("Bot is now running...")
    application.run_polling(poll_interval=10)

if __name__ == "__main__":
    run_bot()
