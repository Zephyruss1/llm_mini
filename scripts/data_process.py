import subprocess
import os
import tiktoken # type: ignore
import requests

def download_data(FILE_PATH):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
    if not os.path.exists(FILE_PATH):
        try:
            txt_file = subprocess.run(["wget", "-O", FILE_PATH, url])
            print(f"Data downloading... to {FILE_PATH}")
            return txt_file
        except requests.HTTPError.strerror as e:
            print(f"Error: {e}")
    else:
        print("---" * 15)
        print(f"Data already exists at {FILE_PATH}")
        pass

def check_data(FILE_PATH):
    # Check if data file exists
    if not os.path.isfile(FILE_PATH):
        print("Data file does not exist")
    # Check if the data file is empty
    if os.path.getsize(FILE_PATH) == 0:
        print("Data file is empty")

    with open(FILE_PATH, 'r') as f:
        data = f.read()

    print(f"Data length: {len(data)}")
    print("---" * 15)
    return "Data is healthy"

def encode_data(FILE_PATH):
    enc = tiktoken.encoding_for_model('gpt-2')
    encode = lambda x: enc.encode(x)
    end_of_text = enc._special_tokens['<|endoftext|>']
    assert enc.name != 'gpt-2', "Encoding model is not GPT-2"

    # Adjusted path for the data file
    data_filename = os.path.join(os.path.dirname(FILE_PATH), "input.txt")
    with open(data_filename, "r") as f:
        text = f.read()
    
    sections = text.split("\n\n")
    tokens = []
    for i, s in enumerate(sections):
        tokens.append(end_of_text)
        spad = s + "\n\n" if i != len(sections) - 1 else s
        tokens.extend(encode(spad))
    
    # Ensure the split index is within range
    split_index = min(32768, len(tokens))
    train_tokens = tokens[split_index:]
    val_tokens = tokens[:split_index]

    # Save the tokenized data
    train_filename = os.path.join(os.path.dirname(TRAIN_PATH), "train.txt")
    val_filename = os.path.join(os.path.dirname(VAL_PATH), "val.txt")
    try:
        if not os.path.exists(train_filename):
            with open(train_filename, "w") as f:
                f.write("\n".join([str(t) for t in train_tokens]))
            print(f"Writing {train_filename}")
        if not os.path.exists(val_filename):        
            with open(val_filename, "w") as f:
                f.write("\n".join([str(t) for t in val_tokens]))
            print(f"Writing {val_filename}")
        else:
            print(f"Files already exist: {train_filename} and {val_filename}")
    except IOError as e:
        raise IOError(f"Error writing to file: {e}")
    
    print("---" * 15)
    return train_filename, val_filename

if __name__ == "__main__":
    TRAIN_PATH  = os.path.join(os.getcwd(), '../data', "train.txt")
    VAL_PATH    = os.path.join(os.getcwd(), '../data', "val.txt")
    FILE_PATH   = os.path.join(os.getcwd(), '../data', 'input.txt')
    # Downloading Data
    download_data(FILE_PATH)
    # Checking Data Healthy
    check_data(FILE_PATH)
    # Encoding Data
    encode_data(FILE_PATH)