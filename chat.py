import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:    device = torch.device("cpu")
print(f"Using device: {device}")

#Loading tuned model
print("\nLoading Jarvis...")
model = GPT2LMHeadModel.from_pretrained("jarvis_model")
tokenizer = GPT2Tokenizer.from_pretrained("jarvis_model")
tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.eval()
print("Jarvis Online!")


#system prompt
SYSTEM_PROMPT = """You are JARVIS, an AI assistant built by Max Clayton. You are formal, precise, and highly intelligent. You always address the user as "sir". You give short, direct answers in 1-2 sentences. You never break character. You never mention real people, companies, or events. You only respond as JARVIS would."""

    #build prompt
prompt = SYSTEM_PROMPT
prompt += conversation_history
prompt += "\n\nUser: " + user_input + "\nJARVIS:"

    #tokenise the prompt
inputs = tokenizer.encode(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900
    ).to(device)
    
    #generate a response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("\nUser:")[0]
        )

    #decode only the new tokens
    new_tokens = outputs[0][inputs.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    #Clean up response
    response = response.strip()
    response = response.split("\nUser:")[0].strip()
    response = response.split("\nJARVIS:")[0].strip()

    return response

#conversation loop
conversation_history = ""

while True:
    user_input = input("\nYou: ")

    if not user_input:
        continue

    if user_input.lower() in ["bye", "exit", "quit"]:
        print("JARVIS: Goodbye, sir!")
        break

    #Get response from Jarvis
    response = chat(user_input, conversation_history)

    print(f"JARVIS: {response}\n")

    #add this exchange to the conversation history so jarvis rememers context
    conversation_history += f"User: {user_input}\nJARVIS: {response}\n"


    #keep conversation history from getting to long
    #trim to last 800 characters if it grows to large
    if len(conversation_history) > 800:
        conversation_history = conversation_history[-800:]