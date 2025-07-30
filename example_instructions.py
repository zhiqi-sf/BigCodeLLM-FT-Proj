# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions = [
        [
            {
                "role": "user",
                "content": "In Bash, how do I list all text files in the current directory (files with .txt extension) that have been modified in the last 24 hours?",
            }
        ],
        [
            {
                "role": "user",
                "content": "What is the difference between inorder and preorder traversal? Give an example in Python.",
            }
        ],
        [
            {
                "role": "user",
                "content": "Write a function in JavaScript that takes an array of integers and returns a new array with only the prime numbers.",
            }
        ],
        [
            {
                "role": "user",
                "content": "In Python, write a function that takes a pandas DataFrame and returns the sum of a specific column. Make sure to handle the case where the column does not exist.",
            }
        ],
    ]
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=None,
        temperature=0.2,
        top_p=0.9,
    )

    for instruction, result in zip(instructions, results):
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)