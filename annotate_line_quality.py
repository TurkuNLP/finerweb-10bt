import json
import os
import random
import re
import time
import warnings

import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from tqdm import tqdm

SAVE_PATH = "llm_line_annotations"


def num_tokens_from_string(speech):
    """Return the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    num_tokens = len(encoding.encode(speech))
    return num_tokens


def generate(input, non_quality_labels):
    """Generate classifications for given lines."""
    if len(non_quality_labels) == 0:
        non_quality_labels = (
            "The list is currently empty. You are free to create new labels."
        )
    else:
        non_quality_labels = "\n".join(non_quality_labels)

    system = "You are an expert text classifier specializing in LLM training data. Your task is to classify each line of text based on its suitability for inclusion in a language model training dataset. High-quality content is clean, meaningful, well-structured, and useful for training language models. Low-quality content includes boilerplate elements (e.g., navigation menus, footers), non-linguistic symbols, formatting tags, placeholders like 'Lorem ipsum', and spammy, irrelevant, or toxic language."

    prompt = f"""
**Instructions:**

1. **Line Identification and Separation**:
   - Each line starts with "Line X:" where X is the line number. Treat each "Line X:" as a single unit, regardless of length; do not split lines.
   - Lines are separated by newline characters (`\n`) and dashes (`------`). If there's no newline character, treat the entire text as a single line.

2. **Contextual Classification**:
   - Use the context of all lines when classifying each one, as they are sequential and from the same document.
   - For example, a line starting with a hyphen might be part of a list and should be classified as "Clean."

3. **Assigning Labels**:
   - Assign **exactly one label** to each line.
   - If the line is suitable for inclusion, label it **"Clean"**.
   - If not, assign a specific and descriptive label explaining why it's unsuitable.
   - **Prefer labels from the provided list**. Only create a new label (max three words) if absolutely necessary.
   - **Do not use vague labels** like "Low-Quality," "Bad," "Unsuitable," etc. Labels must be specific and descriptive.

4. **Focus on Linguistic Content**:
   - Retain valuable and diverse linguistic content suitable for language model pre-training, including natural language patterns, standard advertising copy, commercial language, and promotional content written in natural language.

5. **Tolerance for Minor Errors and Toxic Language**:
   - Minor grammatical errors, typos, or small mistakes do not disqualify a line from being "Clean." Only exclude lines with pervasive errors that significantly hinder understanding.
   - Mild expletives and controversial opinions do not disqualify a line from being "Clean." Only exclude lines with blatantly hateful, harmful or toxic content.

6. **Output Format**:
   - Your output must have exactly the same number of lines as the input, matching each line number correctly.
   - Output only the line number followed by the label, separated by a colon.
   - Do not include any additional text or explanations.
   - Do not output dashes between the lines.

**Guidelines for "Clean" Lines**:

Assign "Clean" to lines that:

- Represent natural language suitable for training language models.
- Include informal internet language, grammatical errors, questions, partial sentences, and common online expressions.
- Contain standard advertising or commercial language in natural sentences.
- Have properly formatted titles, headings, and readable content, even with stylistic elements.
- Include minor in-text elements like email addresses, dates, or URLs within natural sentences.
- Are general promotional content written in natural language.

**Guidelines for Non-"Clean" Lines**:

Lines not classified as "Clean" need a specific and descriptive label. Examples include lines that:

- Contain blatantly hateful or harmful language. 
- Are long passages of non-English text (excluding common foreign phrases used in English).
- Include disclaimers, copyright notices, terms, and conditions.
- Consist of menu items, login links, buttons, or navigation menus.
- Contain random characters, garbled text, or excessive symbols.
- Include programming code, HTML tags, or markup languages (when actual code or markup appears).
- Present keywords, tags, or similar data without sufficient context.
- Are irrelevant or spam-like content not suitable for training.
- Are **excessively** promotional without natural language structure (e.g., a list of product names and prices without sentences).

**Possible Labels for Non-"Clean" Lines**:

{non_quality_labels}

**Example Input:**

Line 1: Welcome to our website!
------
Line 2: Contact us at support@example.com.
------
Line 3: ***** $$$$$
------
Line 4: <div>Content</div>
------

**Example Output:**

Line 1: Clean  
Line 2: Clean  
Line 3: Encoding Errors  
Line 4: HTML Tags

**Now, classify the following lines:**

{input}
"""

    def completion():
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,  # max length of response
        )
        return response

    try:
        response = completion()
    except OpenAIError as e:
        print("Something went wrong with OpenAI. Trying again in 5 seconds...")
        time.sleep(5)
        response = completion()

    return response.choices[0].message.content, system + prompt


def get_key():
    """Get OpenAI authorization key."""
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")


def format_input(input_lines):
    """Format input lines."""
    formatted_lines = ""
    for i, line in enumerate(input_lines):
        formatted_lines += f"*Line {i+1}:* {line}\n------\n"
    return formatted_lines.strip("\n")


def calculate_cost(input_len, output_len):
    """Calculate the cost of running the model based on the input and output tokens."""
    input_price = 0.15 / 1_000_000
    output_price = 0.6 / 1_000_000
    print(
        f"Input tokens: {sum(input_len)} at a cost of ${input_price * sum(input_len)}"
    )
    print(
        f"Output tokens: {sum(output_len)} at a cost of ${output_price * sum(output_len)}"
    )
    print(
        f"Total cost: ${sum([output_price * sum(output_len), input_price * sum(input_len)])}"
    )
    print()


def iterate_in_chunks(doc, batch_size=15):
    """Split document into max batch_size lines. If doc is not divisible into batch_size chunks,
    it is split into even chunks."""
    n = len(doc)
    if n <= batch_size:
        yield doc
    else:
        num_batches = (
            n // batch_size
            if n % batch_size == 0
            else (n + batch_size - 1) // batch_size
        )
        min_batch_size = n // num_batches
        extra_items = n % num_batches

        start = 0
        for i in range(num_batches):
            current_batch_size = min_batch_size + (1 if i < extra_items else 0)
            yield doc[start : start + current_batch_size]
            start += current_batch_size


def extract_junk_labels(junk_labels, output):
    """Extract junk labels from output and add them to the junk list."""
    output_lines = output.split("\n")
    given_labels = [label.split(":", 1)[1].strip() for label in output_lines]
    for label in given_labels:
        junk_labels.append(label)

    junk_labels = [
        label.lower().strip().strip("\n") for label in junk_labels
    ]  # remove trailing spaces etc.
    junk_labels = list(set(junk_labels))  # remove duplicates
    random.shuffle(junk_labels)  # randomize order of labels
    junk_labels = [
        label for label in junk_labels if label != "Clean"
    ]  # remove Clean from junk labels
    junk_labels = [
        label for label in junk_labels if label != "clean"
    ]  # remove clean from junk labels
    return junk_labels


def split_long_line_into_segments(text, batch_size):
    """
    Some documents contain only one, often long, line, which can cause issues.
    This function splits long lines into smaller segments.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)

    segments = []
    current_segment = ""

    for sentence in sentences:
        # If adding the sentence would exceed 200 chars, start a new segment
        if len(current_segment) + len(sentence) + 1 > 200:
            segments.append(
                current_segment.strip()
            )  # Strip to remove any trailing spaces
            current_segment = sentence
        else:
            current_segment += " " + sentence

    # Add the last segment if it's non-empty
    if current_segment:
        segments.append(current_segment.strip())

    return segments


def verify_output(chunk, output):
    """Verify that output matches required formatting."""
    for line in output.splitlines():
        pattern = r"^Line ([1-9]|1[0-9]|2[0-9]):.+"  # Pattern of valid output line
        if not re.match(pattern, line):
            return False
    return True


def main(
    start_from_index=0,
    stop_at_index=500,
    batch_size=10,
    load_junk_labels=False,
    save_file="junk_classification_output.jsonl",
):

    if start_from_index == 0 and load_junk_labels:
        warnings.warn("Using previously saved junk labels!", UserWarning)
    if start_from_index > 0 and not load_junk_labels:
        warnings.warn("Not using previously saved junk labels!", UserWarning)

    get_key()  # load OpenAI API key
    input_len = []
    output_len = []
    time_taken = []

    docs = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
    )

    if load_junk_labels:
        with open(f"{SAVE_PATH}/junk_labels.txt", "r") as f:
            junk_labels = f.readlines()
    else:
        junk_labels = []

    for doc_index, doc in tqdm(enumerate(docs)):
        start_time = time.time()
        if doc_index < start_from_index:
            continue

        doc_output = []
        lines = doc["text"].splitlines()

        # If there is only one line in the doc, it is typically very long.
        # Let's split it into smaller chunks to make thing easier for the model.
        if len(lines) == 1 and len(lines[0]) > 200:
            lines = split_long_line_into_segments(lines[0], batch_size)
            was_split = True
        else:
            was_split = False
        for chunk in iterate_in_chunks(lines, batch_size):
            retries = 0
            while True:
                # Format input.
                input = format_input(chunk)

                # Generate response.
                output, full_prompt = generate(input, junk_labels)

                # Calculate input and output tokens to keep track of costs.
                input_len.append(num_tokens_from_string(full_prompt))
                output_len.append(num_tokens_from_string(output))

                # Verify output formatting and retry if not okay.
                output_is_ok = verify_output(chunk, output)
                if output_is_ok:
                    break
                else:
                    print("Output formatted incorrectly. Retrying...")
                    retries += 1
                    if retries >= 3:
                        raise Exception(
                            f"Too many retries! Failing output:\n{output} at index {doc_index}.\nFormatted input:\n{input}\nRaw data:{chunk} with length {len(chunk)}."
                        )

            # Add generated junk labels to junk_labels list
            junk_labels = extract_junk_labels(junk_labels, output)

            for input_line, output_line in zip(chunk, output.splitlines()):
                dict = {
                    "line": input_line,
                    "label": output_line.split(":")[1]
                    .strip()
                    .lower(),  # Remove the "Line X:" preamble
                    "split": was_split,  # whether the doc was split "manually".
                }
                doc_output.append(dict)

        # Save output.
        with open(f"{SAVE_PATH}/{save_file}", "a") as f:
            dict = {"doc": doc, "content": doc_output}
            f.write(json.dumps(dict, ensure_ascii=False))
            f.write("\n")

        with open(f"{SAVE_PATH}/junk_labels.txt", "w") as f:
            for line in junk_labels:
                f.write(line)
                f.write("\n")

        # Keep track of time to get average time per document.
        end_time = time.time()
        time_taken.append(end_time - start_time)

        # Print cost every now and then while running to make sure we're not bleeding money.
        # Also print the junk labels and how many labels there are to keep an eye on them, too.
        if doc_index > 0 and doc_index % 100 == 0:
            with open(f"{SAVE_PATH}/number_of_labels.csv", "a") as f:
                f.write(f"{doc_index}, {len(junk_labels)}\n")
            calculate_cost(input_len, output_len)
            print(f"Junk labels: {junk_labels}")
            print(f"Number of labels: {len(junk_labels)}")
            print()

        if doc_index >= stop_at_index:
            break

    calculate_cost(input_len, output_len)
    print(
        f"Average time taken to generate labels for one document: {round(sum(time_taken)/len(time_taken), 3)} seconds"
    )


if __name__ == "__main__":
    main(
        start_from_index=0,
        stop_at_index=20_000,
        load_junk_labels=False,
        save_file=f"{SAVE_PATH}/test.jsonl",
    )
