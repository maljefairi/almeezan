import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset
import json
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Arabic BERT model and tokenizer
model_name = "aubmindlab/bert-base-arabertv2"
logger.info(f"Loading model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load sentence transformer for vectorization
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Function to read and preprocess data
def read_data(folders):
    data = []
    for folder in folders:
        logger.info(f"Reading data from folder: {folder}")
        for filename in tqdm(os.listdir(folder), desc=f"Processing {folder}"):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                    data.append({"filename": filename, "content": content})
    return data

# Prepare dataset
folders = ["LawView", "OpinionView", "RulingView", "ViewAgreement"]
logger.info("Preparing dataset")
raw_data = read_data(folders)

# Vectorize documents
def vectorize_documents(data):
    logger.info("Vectorizing documents")
    vectors = []
    for item in tqdm(data, desc="Vectorizing"):
        vector = sentence_model.encode(item["content"])
        vectors.append(vector)
    return np.array(vectors)

document_vectors = vectorize_documents(raw_data)

# Create graph based on document similarity
def create_similarity_graph(vectors, threshold=0.5):
    logger.info("Creating similarity graph")
    similarity_matrix = cosine_similarity(vectors)
    G = nx.Graph()
    for i in range(len(vectors)):
        G.add_node(i, content=raw_data[i]["content"])
        for j in range(i+1, len(vectors)):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i][j])
    return G

similarity_graph = create_similarity_graph(document_vectors)

# Visualize graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(similarity_graph)
nx.draw(similarity_graph, pos, node_size=20, node_color='lightblue', with_labels=False)
plt.title("Document Similarity Graph")
plt.savefig("document_similarity_graph.png")
plt.close()

# Create examples for training using graph-based context
def create_examples_with_context(graph, data):
    examples = []
    logger.info("Creating examples for training with graph-based context")
    for node in tqdm(graph.nodes(), desc="Creating examples"):
        context = data[node]["content"]
        neighbors = list(graph.neighbors(node))
        if neighbors:
            # Add content from a random neighbor for additional context
            random_neighbor = np.random.choice(neighbors)
            context += "\n\n" + data[random_neighbor]["content"]
        
        # Generate a simple question (this should be improved for real-world use)
        question = "ما هي النقاط الرئيسية في هذا النص؟"
        
        examples.append({
            "context": context,
            "question": question,
            "answer": "يرجى استخراج النقاط الرئيسية من النص",
            "filename": data[node]["filename"]
        })
    return examples

train_examples = create_examples_with_context(similarity_graph, raw_data)

# Convert to Dataset object
logger.info("Converting to Dataset object")
dataset = Dataset.from_dict({
    "context": [ex["context"] for ex in train_examples],
    "question": [ex["question"] for ex in train_examples],
    "answer": [ex["answer"] for ex in train_examples],
    "filename": [ex["filename"] for ex in train_examples]
})

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

logger.info("Tokenizing dataset")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Set up training arguments
logger.info("Setting up training arguments")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)

# Initialize Trainer with custom compute_loss function
def compute_loss(model, inputs):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    start_logits, end_logits = outputs.start_logits, outputs.end_logits
    loss_fct = torch.nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, labels[:, 0])
    end_loss = loss_fct(end_logits, labels[:, 1])
    return (start_loss + end_loss) / 2

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    compute_loss=compute_loss
)

# Train the model
logger.info("Starting model training")
trainer.train()

# Save the fine-tuned model
logger.info("Saving the fine-tuned model")
model.save_pretrained("./fine_tuned_arabic_qa_model")
tokenizer.save_pretrained("./fine_tuned_arabic_qa_model")

# Save document vectors for future use
np.save("document_vectors.npy", document_vectors)

# Save graph for future use
nx.write_gpickle(similarity_graph, "similarity_graph.gpickle")

logger.info("Model training completed, vectorization and graph creation finished.")
