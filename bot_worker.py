# bot_worker.py
"""
TruthPulse Telegram worker
- Uses python-telegram-bot v20+ async API
- Integrates agents.verifier.verify_claim(...)
- Safe long message sending (chunks or .txt fallback)
- Robust error handling & logging
"""

import os
import re
import io
import asyncio
import logging
from typing import Optional, List

from dotenv import load_dotenv
from telegram import Update, Bot, error as tg_error
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, CallbackContext

# Import verifier
# Ensure agents package is importable (same project)
from agents.verifier import verify_claim

# Load .env if present
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s"
)
logger = logging.getLogger("bot_worker")

# Config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_MAX_CHARS = 4000  # safe margin under Telegram's ~4096 limit
CHUNK_THRESHOLD = 8  # if more than this chunks, send as file

# Helper: split text into safe chunks preserving paragraphs / sentences
def split_text_into_chunks(text: str, limit: int = TELEGRAM_MAX_CHARS) -> List[str]:
    if not text:
        return []
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            # keep a blank line if small
            if len(current) + 2 < limit:
                current += "\n\n"
            continue
        if len(p) >= limit:
            # paragraph too large; split by sentences
            sentences = re.split(r'(?<=[\.\?\!])\s+', p)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(current) + len(s) + 2 < limit:
                    current = (current + "\n\n" + s).strip() if current else s
                else:
                    if current:
                        chunks.append(current.strip())
                    # if sentence itself too big, break hard
                    if len(s) < limit:
                        current = s
                    else:
                        for i in range(0, len(s), limit - 50):
                            chunks.append(s[i:i + limit - 50].strip())
                        current = ""
        else:
            if len(current) + len(p) + 2 < limit:
                current = (current + "\n\n" + p).strip() if current else p
            else:
                if current:
                    chunks.append(current.strip())
                current = p
    if current:
        chunks.append(current.strip())
    return chunks

# Helper: send long text gracefully (chunks or file fallback)
async def safe_send_long_text(bot: Bot, chat_id: int, text: str, reply_to_message_id: Optional[int] = None, disable_web_page_preview: bool = True, parse_mode = None):
    if not text:
        return []
    # If small enough — send single message
    try:
        if len(text) <= TELEGRAM_MAX_CHARS:
            msg = await bot.send_message(chat_id=chat_id, text=text, reply_to_message_id=reply_to_message_id, disable_web_page_preview=disable_web_page_preview, parse_mode=parse_mode)
            return [msg]
    except tg_error.BadRequest as e:
        logger.debug("single-message send failed: %s", e)
    except Exception as e:
        logger.exception("unexpected error sending single message: %s", e)

    chunks = split_text_into_chunks(text, TELEGRAM_MAX_CHARS)
    if not chunks:
        return []

    # If too many chunks — prefer file
    if len(chunks) > CHUNK_THRESHOLD:
        try:
            buf = io.BytesIO()
            buf.write(text.encode("utf-8"))
            buf.seek(0)
            filename = "truthpulse_report.txt"
            msg = await bot.send_document(chat_id=chat_id, document=buf, filename=filename, caption="Full TruthPulse report (attached).")
            return [msg]
        except Exception as e:
            logger.exception("file fallback failed: %s", e)
            # fall back to chunk sending attempt below

    sent_messages = []
    try:
        for idx, c in enumerate(chunks):
            header = "" if idx == 0 else f"(cont. {idx+1}/{len(chunks)})\n\n"
            part_text = header + c
            msg = await bot.send_message(chat_id=chat_id, text=part_text, disable_web_page_preview=disable_web_page_preview, parse_mode=parse_mode)
            sent_messages.append(msg)
        return sent_messages
    except tg_error.BadRequest as e:
        logger.warning("chunked send failed (BadRequest): %s — falling back to file", e)
    except Exception as e:
        logger.exception("chunked send failed: %s", e)

    # Last resort: file upload
    try:
        buf = io.BytesIO()
        buf.write(text.encode("utf-8"))
        buf.seek(0)
        filename = "truthpulse_report.txt"
        msg = await bot.send_document(chat_id=chat_id, document=buf, filename=filename, caption="Full TruthPulse report (attached).")
        return [msg]
    except Exception as e:
        logger.exception("final file fallback failed: %s", e)
        return []

# Start command handler
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Hello! This is TruthPulse — send a claim or a link and I'll try to verify it.\n\n"
        "Examples:\n"
        "- India, Russia Defence Ministers to meet, S-500 acquisition on agenda\n"
        "- Cyclone Ditwah flights and trains cancelled\n\n"
        "I will reply with a structured fact-check report. (If the report is long I'll send it as chunks or a file.)"
    )
    await update.message.reply_text(text)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "TruthPulse Help:\n"
        "- Send any claim or URL and I'll search multiple sources and return a fact-check summary.\n"
        "- If the reply is very long I'll split it or attach a file.\n"
        "- For admin debugging, set VERBOSE=1 and restart the server."
    )
    await update.message.reply_text(text)

# Core message handler
# Replace your existing handle_message with this function

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Improved handler: reliably sends a placeholder using context.bot, shows typing,
    then edits or sends long text via safe_send_long_text.
    """
    if update.message is None or update.message.text is None:
        try:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please send text or a link to verify.")
        except Exception:
            logger.exception("Failed to notify user to send text.")
        return

    user_text = update.message.text.strip()
    chat_id = update.effective_chat.id
    user = update.effective_user
    logger.info("Received claim from %s (%s): %s", user.full_name if user else "unknown", getattr(user, "id", "unknown"), user_text[:200])

    # Send placeholder reliably via bot (not reply_text)
    sent = None
    try:
        sent = await context.bot.send_message(chat_id=chat_id, text="🔎 Checking — please wait...", disable_web_page_preview=True)
    except Exception as e:
        # Log and continue — we can still send final result
        logger.warning("Failed to send placeholder via bot.send_message: %s", e)
        sent = None

    # Show "typing" action while processing (nice UX)
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    except Exception:
        pass

    # Run verification
    try:
        verdict_text = await verify_claim(user_text)
    except Exception as e:
        logger.exception("Error during verify_claim: %s", e)
        verdict_text = "Error: Something went wrong while verifying the claim. Please try again later."

    # Try to edit the placeholder; if not possible, delete placeholder and send result
    try:
        if sent:
            try:
                if len(verdict_text) <= TELEGRAM_MAX_CHARS:
                    await sent.edit_text(verdict_text, disable_web_page_preview=True)
                    return
            except Exception as e:
                # edit failed (maybe too long or not editable)
                logger.info("Could not edit placeholder: %s", e)
            # Delete placeholder if present (best-effort)
            try:
                await sent.delete()
            except Exception:
                pass
            await safe_send_long_text(context.bot, chat_id, verdict_text, disable_web_page_preview=True)
        else:
            # No placeholder: send the result safely
            await safe_send_long_text(context.bot, chat_id, verdict_text, disable_web_page_preview=True)
    except Exception as e:
        logger.exception("Failed to deliver verdict text: %s", e)
        try:
            await context.bot.send_message(chat_id=chat_id, text="Failed to deliver report. Try again later.")
        except Exception:
            logger.exception("Even last-resort message failed")


# Error handler for application-level errors
async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled application error: %s", context.error)
    try:
        if isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="An internal error occurred. Please try again later.")
    except Exception:
        logger.exception("Failed to send error message to user")

# Build and run bot (call this from your FastAPI startup or run directly)
async def run_bot():
    """
    Build and start the Telegram Application. Designed to be launched as an asyncio task.
    Example usage in FastAPI startup:
        asyncio.create_task(run_bot())
    """
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN is not set. Set TELEGRAM_TOKEN in env or .env.")
        return

    logger.info("Starting Telegram bot...")
    try:
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()
    except Exception as e:
        logger.exception("Failed to build Telegram Application: %s", e)
        return

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_error_handler(global_error_handler)

    # Run the polling loop in a thread to avoid blocking the caller's event loop
    # run_polling() is blocking, so run in background thread
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, app.run_polling)
    except Exception as e:
        logger.exception("Telegram poller stopped with exception: %s", e)
    finally:
        logger.info("Telegram bot shutting down.")

# If you want to run this worker directly (for dev), you can:
if __name__ == "__main__":
    # Direct run for development: runs the bot until you press Ctrl+C
    if not TELEGRAM_TOKEN:
        logger.error("Please set TELEGRAM_TOKEN in environment or .env before running directly.")
        raise SystemExit(1)
    try:
        # Start the bot (blocking)
        from telegram.ext import Application

        # simpler direct run for dev (blocking)
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).concurrent_updates(True).build()
        application.add_handler(CommandHandler("start", cmd_start))
        application.add_handler(CommandHandler("help", cmd_help))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        application.add_error_handler(global_error_handler)

        logger.info("Running bot in direct mode (CTRL+C to stop)...")
        application.run_polling()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.exception("Bot failed: %s", e)
