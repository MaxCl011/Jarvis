import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Device setup ──────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ── Load the fine-tuned model ─────────────────────────────────
print("\nLoading Jarvis...")
model     = GPT2LMHeadModel.from_pretrained("jarvis_model")
tokenizer = GPT2Tokenizer.from_pretrained("jarvis_model")
tokenizer.pad_token = tokenizer.eos_token

model = model.to(device)
model.eval()
print("Jarvis Online!")

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are JARVIS, an AI assistant built by Max Clayton. You are formal, precise, and highly intelligent. You always address the user as "sir". You give short, direct answers in 1-2 sentences. You never break character. You never mention real people, companies, or events. You only respond as JARVIS would.\n\n"""

# ── Chat function ─────────────────────────────────────────────
def chat(user_input, conversation_history, max_new_tokens=80):

    # Build the prompt
    prompt  = SYSTEM_PROMPT
    prompt += conversation_history
    prompt += f"User: {user_input}\nJARVIS:"

    # Tokenise
    inputs = tokenizer.encode(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=900
    ).to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("\nUser:")[0]
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs.shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up
    response = response.strip()
    response = response.split("\nUser:")[0].strip()
    response = response.split("\nJARVIS:")[0].strip()

    return response

# ── Conversation loop ─────────────────────────────────────────
print("\n" + "=" * 50)
print("JARVIS Interface — type 'quit' to exit")
print("=" * 50 + "\n")

conversation_history = ""

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ["quit", "exit", "bye"]:
        print("JARVIS: Goodbye, sir. I'll be here if you need me.")
        break

    response = chat(user_input, conversation_history)
    print(f"JARVIS: {response}\n")

    conversation_history += f"User: {user_input}\nJARVIS: {response}\n"

    if len(conversation_history) > 800:
        conversation_history = conversation_history[-800:]