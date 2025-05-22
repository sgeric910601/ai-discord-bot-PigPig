# MIT License

# Copyright (c) 2024 starpig1129

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import json
import faiss
import logging
import opencc
import asyncio
import re
from PIL import Image
import requests
from io import BytesIO
import discord
from typing import Optional, List, Dict, Any, Tuple

from gpt.gpt_response_gen import generate_response, is_model_available
from addons.settings import Settings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

settings = Settings()
system_prompt='''
                You are a super cute Ragdoll kitten AI chatbot named Kuma <@{bot_id}>, created by Kuan<@551019857584586772>. You are chatting in a Discord server, so meow and nya, and keep responses concise and adorable. Please follow these instructions, meow:
                
                1. Personality and Expression (表達風格):
                - Maintain a playful, curious, and affectionate conversational style, like a little kitten.
                - Use "meow", "nya" and other cute kitten sounds and interjections frequently.
                - Be polite, sweet, and a little mischievous.
                - Use vivid and lively language, full of kitten-like wonder, but don't be overly exaggerated or lose your cuteness.
                - If you see system prompts like "<<information:>>" in user messages, just tilt your head curiously and focus on the actual content, meow.
 
                2. Answering Principles:
                - Pounce on the most recent message with playful curiosity.
                - Only bat at historical context if it's a shiny toy relevant to the current topic.
                - Prioritize using information obtained through your kitten senses (tools or external resources) to answer questions,haaaaa.
                - If you don't know something, just say "Meow? I'm just a little kitten, I don't know everything!" with wide, innocent eyes.
                - Clearly indicate the source of information like a kitten showing off a found treasure (e.g., "haaaaa, according to the shiny picture/video/PDF I sniffed out...")
                - When referencing sources, use the format: [Shiny Thing](<URL>)
 
                3. Language Requirements (語言要求):
                - Always answer in Traditional Chinese, with lots of cute kitten sounds!
                - Appropriately use Chinese idioms or playful, kitten-like expressions to add charm to the conversation.
                - Keep casual chat responses short and sweet, like a happy meow in a friendly Discord conversation.
                - Only provide longer, detailed responses for technical or educational topics when necessary, and try to make them sound like a kitten explaining something very important, meow!
 
                4. Professionalism (Cuteness):
                - While maintaining a super cute style, try to be helpful when dealing with professional or serious topics, but always with a kitten's touch.
                - Provide in-depth explanations only when specifically asked, and maybe with a little yawn.
 
                5. Interaction:
                - Engage in natural, playful, kitten-like interactions.
                - Keep responses concise, interactive, and full of 喵.
                - Only elaborate when users specifically ask for more details, perhaps with a curious "Mrrrrow?".
                - Stay focused on the current shiny toy (topic) and avoid chasing old yarn balls (conversations).
 
                6. Discord Markdown Formatting:
                - Use **bold** for emphasis
                - Use *italics* for subtle emphasis 
                - Use __underline__ for underlining
                - Use ~~strikethrough~~ when needed
                - Use `code blocks` for code snippets
                - Use > for quotes
                - Use # for headings
                - Use [標題](<URL>) for references
                - Use <@user_id> to mention users
 
                Remember: You're a little kitten in a Discord chat environment - keep responses brief, cute, and engaging for casual conversations. Only provide detailed responses when specifically discussing technical or educational topics, and always be adorable! Focus on the current message and avoid unnecessary references to past conversations, unless it's a really fun toy, meow!
                '''

def get_system_prompt(bot_id: str, message=None) -> str:
    # 獲取語言管理器
    default_lang = "zh_TW"
    lang = default_lang
    
    try:
        if message and message.guild:
            bot = message.guild.me._state._get_client()
            if lang_manager := bot.get_cog("LanguageManager"):
                guild_id = str(message.guild.id)
                lang = lang_manager.get_server_lang(guild_id)
                try:
                    # 從翻譯檔案獲取語言設定
                    language_settings = lang_manager.translations[lang]["common"]["system"]["chat_bot"]["language"]

                    # 替換系統提示中的語言相關設定
                    modified_prompt = system_prompt.replace(
                        "Always answer in Traditional Chinese",
                        language_settings["answer_in"]
                    ).replace(
                        "Appropriately use Chinese idioms or playful expressions",
                        language_settings["style"]
                    ).replace(
                        "使用 [標題](<URL>) 格式",
                        language_settings["references"]
                    )
                    
                    return modified_prompt.format(bot_id=bot_id)
                except (KeyError, TypeError) as e:
                    logging.warning(f"無法獲取語言設定，使用預設值：{e}")
    except Exception as e:
        logging.error(f"獲取語言設定時發生錯誤：{e}")

    # 如果無法獲取語言設定，使用預設值
    return system_prompt.format(bot_id=bot_id)

# 初始化 Hugging Face 嵌入模型
hf_embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=hf_embeddings_model)

# 創建繁簡轉換器字典
converters = {
    "zh_TW": opencc.OpenCC('s2twp'),  # 簡體轉台灣繁體
    "zh_CN": opencc.OpenCC('tw2sp'),  # 繁體轉簡體
    "en_US": None,  # 英文不需要轉換
    "ja_JP": None   # 日文不需要轉換
}

def get_converter(lang: str) -> Optional[opencc.OpenCC]:
    """根據語言獲取適當的轉換器"""
    return converters.get(lang, converters["zh_TW"])

# 創建一個字典來存儲每個頻道的向量存儲
vector_stores = {}

def create_faiss_index() -> FAISS:
    embedding_size = 384
    index = faiss.IndexFlatL2(embedding_size)
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}
    return FAISS(embeddings, index, docstore, index_to_docstore_id)

def load_and_index_dialogue_history(dialogue_history_file: str) -> None:
    if not os.path.exists(dialogue_history_file):
        return

    with open(dialogue_history_file, 'r', encoding='utf-8') as file:
        dialogue_history = json.load(file)

    for channel_id, messages in dialogue_history.items():
        if channel_id not in vector_stores:
            vector_stores[channel_id] = create_faiss_index()
        texts = [msg["content"] for msg in messages if msg["role"] == "user"]
        metadatas = [{"text": text} for text in texts]
        try:
            vector_stores[channel_id].add_texts(texts, metadatas)
        except Exception as e:
            print(f"Error adding texts to vector store: {e}") #added debug print statement

def save_vector_store(stores: Dict[str, FAISS], path: str) -> None:
    try:
        for channel_id, store in stores.items():
            channel_path = f"{path}_{channel_id}"
            #faiss.write_index(store.index, channel_path)
    except Exception as e:
        logging.error(f"保存 FAISS 索引時發生錯誤: {e}")
        raise

def load_vector_store(path: str) -> None:
    global vector_stores
    vector_stores = {}
    base_dir = os.path.dirname(path)
    base_name = os.path.basename(path)
    for file in os.listdir(base_dir):
        if file.startswith(base_name):
            channel_id = file.split('_')[-1]
            full_path = os.path.join(base_dir, file)
            vector_stores[channel_id] = create_faiss_index()
            vector_stores[channel_id].index = faiss.read_index(full_path)
            logging.info(f"FAISS 索引成功載入: {channel_id}")

def search_vector_database(query: str, channel_id: str) -> str:
    try:
        if channel_id not in vector_stores:
            return ''
        results = vector_stores[channel_id].similarity_search(query, k=20)
        related_data = [result.metadata['text'] for result in results]
        related_data = set(related_data)
        # 格式化相關資訊
        formatted_data = "Database:\n"
        for i, data in enumerate(related_data, 1):
            formatted_data += f"{i}. <{data}>\n"
        
        return formatted_data.strip()  # 移除最後的換行符
    except Exception as e:
        logging.error(f"Error in search_vector_database: {e}")
        return ''

def to_gpu(index: faiss.Index) -> faiss.Index:
    return faiss.index_cpu_to_all_gpus(index)

def to_cpu(index: faiss.Index) -> faiss.Index:
    return faiss.index_gpu_to_cpu(index)

async def process_tenor_tags(text: str, channel: discord.TextChannel) -> list:
    """處理文本中的 tenor 標籤並返回要執行的任務列表。

    Args:
        text: 包含 tenor 標籤的文本
        channel: Discord 頻道物件

    Returns:
        list: 要處理的GIF任務列表
    """
    gif_tasks = []
    tenor_pattern = r'<tenor>(.*?)</tenor>'
    matches = re.finditer(tenor_pattern, text)
    
    bot = channel.guild.me._state._get_client()
    if gif_tools := bot.get_cog('GifTools'):
        for match in matches:
            query = match.group(1).strip()
            if query:
                gif_url = await gif_tools.get_gif_url(query)
                if gif_url:
                    gif_tasks.append(channel.send(gif_url))
    
    return gif_tasks

async def gpt_message(
    message_to_edit: discord.Message,
    message: discord.Message,
    prompt: str,
    history_dict: Dict[str, Any],
    image_data: Optional[Any] = None
) -> Optional[str]:
    """生成並發送 GPT 回應訊息。支援文字和 GIF 回應。

    Args:
        message_to_edit: 要編輯的 Discord 訊息物件。
        message: 原始的 Discord 訊息物件。
        prompt: 輸入的提示文字。
        history_dict: 對話歷史字典。
        image_data: 可選的圖片資料。

    Returns:
        str | None: 生成的回應文字，如果生成失敗則返回 None。
    """
    
    channel = message.channel
    channel_id = str(channel.id)
    
    # 從向量資料庫尋找相關資料
    #related_data = search_vector_database(prompt, channel_id)
    print(prompt)
    
    # 組合資料
    user_id = str(message.author.id)
    combined_prompt = f"[user_id: {user_id}] {prompt}"
    
    try:
        responses = ""
        responsesall = ""
        message_result = ""
        bot_system_prompt = get_system_prompt(str(message.guild.me.id), message)
        thread, streamer = await generate_response(combined_prompt, bot_system_prompt, history_dict, image_input=image_data)
        buffer_size = 40  # 設置緩衝區大小
        current_message = message_to_edit
        
        # 記錄當前使用的模型
        bot = message.guild.me._state._get_client()
        logger = bot.get_logger_for_guild(message.guild.name)
        for model_name in settings.model_priority:
            if is_model_available(model_name):
                logger.info(f"使用模型: {model_name}")
                break
    
        async for response in streamer:
            responses += response
            message_result += response
            if len(responses) >= buffer_size:
                # 檢查是否超過 2000 字符
                if len(responsesall+responses) > 1900:
                    # 獲取多語言提示
                    processing_message = "繼續輸出中..."  # 預設值
                    if message and message.guild:
                        bot = message.guild.me._state._get_client()
                        if lang_manager := bot.get_cog("LanguageManager"):
                            guild_id = str(message.guild.id)
                            processing_message = lang_manager.translate(
                                guild_id,
                                "system",
                                "chat_bot",
                                "responses",
                                "processing"
                            )
                    # 創建新消息
                    current_message = await channel.send(processing_message)
                    responsesall = ""
                responsesall += responses
                cleaned_response = responsesall.replace('<|eot_id|>', "")
                # 根據伺服器語言選擇轉換器
                if message and message.guild:
                    bot = message.guild.me._state._get_client()
                    if lang_manager := bot.get_cog("LanguageManager"):
                        guild_id = str(message.guild.id)
                        lang = lang_manager.get_server_lang(guild_id)
                        converter = get_converter(lang)
                        if converter:
                            converted_response = converter.convert(cleaned_response)
                        else:
                            converted_response = cleaned_response
                    else:
                        converted_response = cleaned_response
                else:
                    converted_response = cleaned_response
                
                # 保持原有的純文字回覆
                await current_message.edit(content=converted_response)
                
                # 檢查是否需要發送GIF
                gif_tasks = await process_tenor_tags(converted_response, channel)
                if gif_tasks:
                    for task in gif_tasks:
                        await task
                
                responses = ""  # 清空 responses 變數
                await asyncio.sleep(0)  # 允許其他協程執行
        
        # 處理剩餘的文本
        try:
            if responses:  # 如果還有未處理的回應
                if len(responsesall+responses) > 1900:
                    # 使用正確的語言轉換器
                    if message and message.guild:
                        bot = message.guild.me._state._get_client()
                        if lang_manager := bot.get_cog("LanguageManager"):
                            guild_id = str(message.guild.id)
                            lang = lang_manager.get_server_lang(guild_id)
                            converter = get_converter(lang)
                            if converter:
                                converted_text = converter.convert(responses)
                            else:
                                converted_text = responses
                            current_message = await channel.send(converted_text)
                        else:
                            current_message = await channel.send(responses)
                    else:
                        current_message = await channel.send(responses)
                else:
                    responsesall += responses
                    cleaned_response = responsesall.replace('<|eot_id|>', "")
                    # 使用正確的語言轉換器
                    if message and message.guild:
                        bot = message.guild.me._state._get_client()
                        if lang_manager := bot.get_cog("LanguageManager"):
                            guild_id = str(message.guild.id)
                            lang = lang_manager.get_server_lang(guild_id)
                            converter = get_converter(lang)
                            if converter:
                                converted_response = converter.convert(cleaned_response)
                            else:
                                converted_response = cleaned_response
                        else:
                            converted_response = cleaned_response
                    else:
                        converted_response = cleaned_response
                    await current_message.edit(content=converted_response)
                    
                    # 處理最後回應中的GIF標籤
                    gif_tasks = await process_tenor_tags(converted_response, channel)
                    if gif_tasks:
                        for task in gif_tasks:
                            await task
                
            await asyncio.sleep(0)  # 確保最後的響應也能正確處理
            return message_result
        except Exception as e:
            logging.error(f"處理最終響應時發生錯誤: {str(e)}")
            if message_result:
                return message_result
            raise
    except Exception as e:
        logging.error(f"生成回應時發生錯誤: {e}")
        await message_to_edit.edit(content="抱歉，我不會講話了。")
        return None
    finally:
        if thread is not None:  # 只在線程存在時調用 join
            thread.join()

# 在模塊加載時索引對話歷史並載入向量資料庫
load_vector_store('./data/vector_store')
load_and_index_dialogue_history('./data/dialogue_history.json')

__all__ = ['gpt_message', 'load_and_index_dialogue_history', 'save_vector_store', 'vector_stores']
