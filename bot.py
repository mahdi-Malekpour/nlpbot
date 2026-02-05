# bot.py - ربات کامل تحلیل احساسات فارسی

import telebot
import joblib
import re

# ==================== بخش ۱: توکن ربات ====================
BOT_TOKEN = "8499148067:AAGdFCmMCMY4kFKevmrvMl7dPxj-g17TcKI"  # از @BotFather بگیر

# ==================== بخش ۲: بارگذاری مدل ====================
try:
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    print("✅ مدل‌ها بارگذاری شدند")
except:
    print("❌ فایل مدل پیدا نشد! اول مدل رو ذخیره کن")
    exit()

# ==================== بخش ۳: تابع پاک‌سازی متن ====================
def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text)  # حذف علائم
    text = re.sub(r'\d+', '', text)      # حذف اعداد
    return text.strip()

# ==================== بخش ۴: تابع پیش‌بینی ====================
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    
    # نگاشت نتایج
    feelings = {
        -1: {"emoji": "😠", "text": "منفی", "color": "🔴"},
        0: {"emoji": "😐", "text": "خنثی", "color": "🟡"},
        1: {"emoji": "😊", "text": "مثبت", "color": "🟢"}
    }
    
    return feelings[pred]

# ==================== بخش ۵: ساخت ربات ====================
bot = telebot.TeleBot(BOT_TOKEN)

# ==================== بخش ۶: دستورات ربات ====================
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
    🤖 **ربات تحلیل احساسات فارسی**
    
    ✍️ هر متن فارسی بفرستید، احساس آن را تشخیص می‌دهم!
    
    📌 مثال‌ها:
    - "این محصول عالیه"
    - "خیلی بد بود"
    - "نه خوب بود نه بد"
    
    🎯 برای شروع، یک متن فارسی بفرستید...
    """
    bot.reply_to(message, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['about'])
def about_bot(message):
    about_text = """
    🧠 **درباره این ربات:**
    
    • مدل: تحلیل احساسات سه‌کلاسه
    • زبان: فارسی
    • کلاس‌ها: مثبت 😊 / خنثی 😐 / منفی 😠
    • دقت: ~۷۲٪
    • توسعه‌دهنده: شما! 🚀
    """
    bot.reply_to(message, about_text)

@bot.message_handler(func=lambda message: True)
def analyze_message(message):
    try:
        # پیش‌بینی
        result = predict_sentiment(message.text)
        
        # ساخت پاسخ
        response = f"""
        📊 **تحلیل احساسات:**
        
        متن: `{message.text[:50]}...`
        
        🎯 نتیجه: **{result['text']}** {result['emoji']}
        🎨 رنگ: {result['color']}
        
        ✨ تحلیل: متن شما احساس **{result['text']}** دارد.
        """
        
        bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        bot.reply_to(message, f"❌ خطا در پردازش:\n{str(e)}")

# ==================== بخش ۷: اجرای ربات ====================
if __name__ == "__main__":
    print("🚀 ربات در حال راه‌اندازی...")
    print(f"🔗 لینک ربات: https://t.me/{(bot.get_me()).username}")
    bot.polling()