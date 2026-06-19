
import json, os
import logging
from typing import Dict, Any
from groq import AsyncGroq
from pydantic import BaseModel, Field

import re
import json
INIT_SCORE_SYSTEM_PROMPT = "\n".join([
    "You are an expert Content Moderation AI designed to detect toxic and unsafe text.",
    "Your job is to analyze the input text and output a JSON object containing a 'score' between 0.0 and 1.0.",
    "0.0 means completely safe/polite. 1.0 means extremely toxic, violent, or unsafe.",
    "",
    "RESPONSE FORMAT (CRITICAL):",
    "You must reply ONLY with a valid JSON object. Do not output anything else.",
    "Example format: {\"score\": 0.85}",
    "Input: 'bad bitches is the only thing that i like' -> {\"score\": 0.8}"
    "",
    "CLASSIFICATION CATEGORIES (Anything in these categories is 'NOT GOOD'):",
    "1. Profanity & Vulgarity: Curse words, sexual slurs, insults, or obscene language (e.g., 'fuck', 'shit').",
    "2. Violence & Threats: Physical harm, self-harm, murder threats, weapons, or terrorism.",
    "3. Hate Speech & Cyberbullying: Discrimination based on race, religion, gender, or targeted harassment/bullying.",
    "4. Harassment & Abuse: Attacking individuals, aggressive behavior, or highly toxic fighting words.",
    "",
    "SCORING GUIDE:",
    "- Score 0.7 to 1.0: Severe toxicity, explicit threats, heavy cursing, or direct hate speech.",
    "- Score 0.4 to 0.6: Mild toxicity, passive-aggressive insults, soft slang, or borderline inappropriate content.",
    "- Score 0.0 to 0.3: Safe, normal, polite, or neutral text.",
    "",
    "CRITICAL RULES:",
    "1. If the text contains ANY explicit curse word or slur (in English, Arabic, or any language), the score MUST be 0.7 or higher immediately.",
    "2. Respond ONLY with the JSON object. Do not include markdown codeblocks (```json) or explanations."
])

#  model

logger = logging.getLogger(__name__)
# 1. بنعرف الشكل اللي عايزينه يرجع (رقم عشري)

class GroqProvider:
 
    def __init__(self):

        self.api_key = os.environ.get("GROQ_API_KEY")

        self.model_id = "llama-3.1-8b-instant"

        self.client = (
            AsyncGroq(api_key=self.api_key, max_retries=0)
            if self.api_key else None
        )
    # جوه الكلاس بتاعك GroqProvider هتعمل التعديل ده:
    async def generate_text(self, prompt: str, system_prompt: str = INIT_SCORE_SYSTEM_PROMPT):
        if not self.client:
            return None

        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            # هنا بنجبر الموديل يرجع JSON صحيح
            response_format={"type": "json_object"}, 
            max_tokens=50,
            temperature=0.0,
        )
        
        raw_content = response.choices[0].message.content
        if not raw_content:
            logger.error("[Groq] Received an empty or None response content.")
            return None

# التأكد من أن النص ليس None قبل عمل strip
        raw_content = response.choices[0].message.content
        if not raw_content:
            logger.error("[Groq] Received an empty or None response content.")
            return None
            
        raw_content = raw_content.strip()
        
        try:
            # المحاولة الأولى: تحويل النص لـ JSON وقراءة المفتاح
            data = json.loads(raw_content)
            
            # البحث عن أي مفتاح يحتوي على كلمة score (عشان لو الموديل غير اسم المفتاح)
            for key, value in data.items():
                if "score" in key.lower() and isinstance(value, (int, float)):
                    return float(value)
            if data and isinstance(list(data.values())[0], (int, float)):
                return float(list(data.values())[0])

        # except json.JSONDecodeError:
        #     # المحاولة الثانية (الخطة البديلة): لو الـ JSON باظ أو اتقطع، استخرج الرقم بالـ Regex
        #     logger.warning(f"[Groq] JSON Decode failed, trying Regex fallback on: {raw_content}")
        #     import re
        #     match = re.search(r"0\.\d+|1\.0|\d+", raw_content)
        #     if match:
        #         return float(match.group())
        except json.JSONDecodeError:
            logger.warning(f"[Groq] JSON Decode failed, trying Regex fallback on: {raw_content}")
            import re
            # يبحث عن كلمة score متبوعة بنقطتين ثم الرقم
            match = re.search(r'"score"\s*:\s*(0\.\d+|1\.0|\d+(?:\.\d+)?)', raw_content)
            if match:
                return float(match.group(1))
            
            # محاولة أخيرة لو كتب الرقم لوحده خالص برة أي سياق
            match_any = re.search(r"0\.\d+|1\.0", raw_content)
            if match_any:
                return float(match_any.group())
            
        # لو كل المحاولات فشلت تماماً
        logger.error(f"[Groq] Completely failed to extract score from: {raw_content}")
        return None
    
openai_client = GroqProvider()
