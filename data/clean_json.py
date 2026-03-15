import json

def get_plain_text(msg):
    """
    Extract clean text from one Telegram message dict.
    Handles both plain string and list-of-pieces format.
    """
    raw = msg.get("text", "")
    
    if isinstance(raw, str):
        return raw.strip()
    
    if not isinstance(raw, (list, tuple)):
        return ""
    
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            t = item.get("type")
            if t in ("text", "plain", "link", "text_link", "mention", 
                     "hashtag", "cashtag", "bot_command", "blockquote"):
                result.append(item.get("text", ""))
            # Add more types here if needed (pre, code, blockquote, ...)

    return "".join(result).strip()


def extract_all_messages_text(filepath):
    """
    Load Telegram JSON export and return list of cleaned message texts
    (only regular messages that contain text)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

    texts = []
    for msg in data.get("messages", []):
        # Optional: skip service messages, empty texts, etc.
        if msg.get("type") != "message":
            continue
        text = get_plain_text(msg)
        if text:  # skip empty lines
            texts.append(text)
    
    return texts


def save_clean_texts(filepath, output_file="./data/clean_texts.txt"):
    texts = extract_all_messages_text(filepath)
    
    if not texts:
        print("No messages with text found or file could not be read.")
        return
    
    with open(output_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text)
            f.write("\n\n")   # double newline between messages
    
    print(f"Saved {len(texts)} messages to {output_file}")

JSON_PATH = "./data/result.json"
save_clean_texts(JSON_PATH)

def load_clean_texts():
    messages = []
    current = []
    filepath="./data/clean_texts.txt"
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()  # remove trailing \n
            if line == "":        # blank line → end of previous message
                if current:
                    messages.append("\n".join(current).strip())
                    current = []
            else:
                current.append(line)
    
    # Don't forget the last message if file doesn't end with blank line
    if current:
        messages.append("\n".join(current).strip())
    
    # Remove any remaining empty entries
    messages = [msg for msg in messages if msg]
    
    return messages