from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(sequence, max_length):
    model_path = "./ckpts/checkpoint-4000"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained('./ckpts/tokenizer')
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

sequence = input()
max_len = int(input())
generate_text(sequence, max_len)