# RAG-Multimodal-Persuasive-Memes-Detection
RAG Multimodal Persuasive Meme Detection using Visual and Textual Feature

## Persuasive Meme Detection
This project focuses on the detection and classification of persuasive memes. The research is divided into three main tasks:

# Tasks Overview

# Task 1: Persuasive vs Non-Persuasive Meme Classification
The first task involves utilizing dataset variables such as text, image, and processed text data (Rag) to classify memes as either persuasive or non-persuasive. This binary classification step helps filter out non-persuasive memes (label: 0) from those containing persuasive content (label: 1).

# Task 2: Intensity of Persuasion Detection
Upon identifying a meme as persuasive (label: 1) in Task 1, the next step is to determine the intensity of persuasion using dataset variables like persuasive_inten. This variable categorizes the intensity into six levels (0 to 5), representing the degree of persuasion from None to Slightly Positively persuasive

0: None <br>
1: Negatively persuasive <br>
2: Slightly Negatively persuasive <br>
3: Neutral <br>
4: Positively persuasive <br>
5: Slightly Positively persuasive <br>

# Task 3: Type of Persuasion Detection
For persuasive memes (label: 1) categorized by intensity in Task 2, Task 3 involves detecting the specific type of persuasive technique used. Dataset variables such as None, Negatively persuasive, Slightly Negatively persuasive, Neutral, Positively persuasive, and Slightly Positively persuasive represent different types of persuasive techniques employed in memes. The task uses these variables to identify and categorize the type of persuasion technique used in each persuasive meme.

1. Personification <br>
2. Irony <br>
3. Alliteration <br>
4. Analogies <br>
5. Invective <br>
6. Metaphor <br>
7. Puns and Wordplays <br>
8. Satire <br>
9. Hyperboles <br>
