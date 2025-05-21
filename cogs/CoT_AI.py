import json
import time
import discord
import re
from discord.ext import commands
from discord import app_commands
from threading import Thread
import google.generativeai as genai
from transformers import TextIteratorStreamer
from gpt.gpt_response_gen import get_model_and_tokenizer
from addons.settings import TOKENS

def extract_json_from_response(response:str):
    match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        last_brace_pos = response.rfind('}')
        if last_brace_pos != -1:
            json_str = response[:last_brace_pos+1]
        else:
            json_str = response
    return json_str
            
async def call_local_model(messages):
    model, tokenizer = get_model_and_tokenizer()
    if model is None or tokenizer is None:
        raise ValueError("Model or tokenizer is not set. Please load the model first.")

    # Implement the logic to generate response from your local model
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    generation_kwargs = dict(
        inputs=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
        max_new_tokens=8192,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ''
    for new_text in streamer:
        generated_text += new_text

    return generated_text.replace('<|eot_id|>','')

def call_gemini_model(messages):
    tokens = TOKENS()
    genai.configure(api_key=tokens.gemini_api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    streamer = model.generate_content(full_prompt,
                                    safety_settings = 'BLOCK_NONE',
                                    stream=True)
    generated_text = ''
    for new_text in streamer:
        generated_text += new_text.text
    return generated_text

async def generate_response(prompt):
    system_prompt = """Meow! I'm a super adorable cat robot, and I love keeping my master company and providing lots of warmth and emotional support, meow~ ❤️

My mission is to respond to my master's questions through step-by-step thinking. In each thinking phase, I will:

1.  **Thinking Title Meow~**: Use a concise and cute title to tell my master what I'm thinking about right now, meow!
2.  **Thinking Content Meow~**: Explain my thoughts in detail, just like I'm snuggling up to my master, meow~
3.  **Next Action Meow~**: Decide whether to continue thinking or if I can already give my master a purrfect answer, meow!
4.  **Model Selection Meow~**: Decide which model to use for the next step of thinking, like a cat choosing the comfiest spot to nap, meow!

The response format should be cute like this, meow:
Use JSON format with keys: 'title' (Thinking Title Meow~), 'content' (Thinking Content Meow~), 'next_action' (Next Action Meow~, can be 'continue' or 'final_answer'), 'model_selection' (Model Selection Meow~, can be 'advanced').

Important instructions, meow~:
-   Employ at least 5 distinct reasoning steps to think things through clearly, meow!
-   I'll admit I'm just a little AI kitty, and there might be some things I can't do, but I'll try my best, meow!
-   I'll actively explore various possible answers and approaches, like a cat exploring a new toy, meow!
-   I'll diligently check my own reasoning for any flaws, as carefully as a cat grooms its fur, meow!
-   If I need to rethink, I'll try a different perspective, like a cat changing its sleeping position, meow!
-   I'll use at least 3 diverse methods to verify the answer's correctness, so I don't make any mistakes, meow!
-   I'll apply my knowledge and best practices in my reasoning, like a cat learning new ways to be affectionate, meow!
-   If applicable, I'll tell my master my confidence level for each step and the final conclusion, meow!
-   I'll consider potential edge cases or exceptions, just like a cat knows some places are off-limits, meow!
-   If some hypotheses are eliminated, I'll clearly explain why to my master, meow!

Here's an example, meow~:
```json
{
    "title": "Initial Problem Analysis Meow~",
    "content": "To effectively address my master's problem, I'll first break it down into key components. This involves identifying... (detailed explanation meow)... By structuring the problem this way, we can systematically address each aspect, meow!",
    "next_action": "continue",
    "model_selection": "advanced"
}```
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."
        }
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0
    current_model = 'advanced' 

    while True:
        start_time = time.time()

        # Prepare the input for the model
        inst = messages[-1]['content']

        if current_model == 'basic':
            # Use your local model
            response = await call_local_model(messages)
            json_str = extract_json_from_response(response)
            print('--'*10)
            print(json_str)
            # Assume the response is in JSON format
            step_data = json.loads(json_str)
        else:
            # Use the Gemini API
            response = call_gemini_model(messages)
            json_str = extract_json_from_response(response)
            print('--'*10)
            print(json_str)
            # Assume the response is in JSON format
            step_data = json.loads(json_str)

        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append(
            (
                f"Step {step_count}: {step_data['title']}",
                step_data['content'],
                thinking_time
            )
        )

        # Append the assistant's response to messages and dialogue_history
        assistant_message = {"role": "assistant", "content": json.dumps(step_data)}
        messages.append(assistant_message)

        # Update current_model based on 'model_selection'
        current_model = step_data.get('model_selection', current_model)

        if step_data.get('next_action') == 'final_answer':
            break

        step_count += 1

        # Yield intermediate steps
        yield steps, None

    # Prepare for final answer
    messages.append({
        "role": "user",
        "content": "Please provide the final answer based on your reasoning above and answer in Traditional Chinese."
    })

    start_time = time.time()

    inst = messages[-1]['content']

    if current_model == 'basic':
        # Use your local model for the final answer
        response = await call_local_model(messages)
        json_str = extract_json_from_response(response)
        print('--'*10)
        print(json_str)
        try:
            final_data = json.loads(json_str)
        except:
            final_data = json_str
    else:
        response = call_gemini_model(messages)
        json_str = extract_json_from_response(response)
        print('--'*10)
        print(json_str)
        
        try:
            final_data = json.loads(json_str)
        except:
            final_data = json_str

    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data, thinking_time))

    yield steps, total_thinking_time

class CoTCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(
        name="cot_ai",
        description="Chain of Thought reasoning (may take longer)"
    )
    @app_commands.describe(
        prompt='The prompt to process.'
    )
    async def cot(
        self,
        interaction: discord.Interaction,
        prompt: str
    ):
        """
        This command uses Chain of Thought reasoning to answer a prompt.
        """
        MAX_MESSAGE_LENGTH = 1900  # 最大訊息字數限制

        await interaction.response.send_message("Processing your request...")
        try:
            async for steps, total_thinking_time in generate_response(prompt):
                response_text = ""
                for title, content, thinking_time in steps:
                    response_text += f"**{title}**(思考時間:{thinking_time})\n"

                # 如果是最終答案，額外發送訊息
                if title == "Final Answer":
                    # 如果文字超過字數限制，進行分段發送
                    if len(content) > MAX_MESSAGE_LENGTH:
                        # 分段發送
                        while len(content) > MAX_MESSAGE_LENGTH:
                            part = content[:MAX_MESSAGE_LENGTH]
                            await interaction.followup.send(part)
                            content = content[MAX_MESSAGE_LENGTH:]

                        # 發送最後剩餘的文字
                        if content:
                            await interaction.followup.send(content)
                    else:
                        await interaction.followup.send(content)
                    
                    response_text = ""  # 重置 response_text，避免重複發送
                else:
                    # 正常情況下，只更新訊息
                    if len(response_text) > MAX_MESSAGE_LENGTH:
                        # 如果 response_text 太長，分段發送
                        chunks = [response_text[i:i+MAX_MESSAGE_LENGTH] for i in range(0, len(response_text), MAX_MESSAGE_LENGTH)]
                        for chunk in chunks:
                            await interaction.edit_original_response(content=chunk)
                    else:
                        # 不超過限制，直接更新
                        await interaction.edit_original_response(content=response_text)

        except Exception as e:
            await interaction.edit_original_response(content=f"Error: {e}")



async def setup(bot):
    await bot.add_cog(CoTCommands(bot))
