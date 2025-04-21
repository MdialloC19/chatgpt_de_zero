from loguru import logger
from pxl_gpt.tokenizer import BPETokenizer


def main():
    tokenizer = BPETokenizer()
    tokenizer.load("data/tokenizer.json")

    logger.info("Testing tokenizer...")
    test_text = "Bonjour, comment ça va?"
    encoded_text = tokenizer.encode(test_text)

    tokens_with_values = tokenizer.get_tokens_with_values(test_text)
    logger.info(f"Tokens with values: {tokens_with_values}")
    logger.info(f"Encoded text {test_text}: {encoded_text}")

    decoded_text = tokenizer.decode(encoded_text)
    logger.info(f"Decoded text: {decoded_text}")


if __name__ == "__main__":
    main()
