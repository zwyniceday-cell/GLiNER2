# GLiNER2 Training and Data Tutorial

Complete guide to preparing data and training GLiNER2 models for entity extraction, classification, structured data extraction, and relation extraction.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Format Overview](#data-format-overview)
3. [Creating Training Examples](#creating-training-examples)
4. [Data Validation](#data-validation)
5. [Training Configuration](#training-configuration)
6. [Running Training](#running-training)
7. [Advanced Topics](#advanced-topics)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Example

```python
from gliner2 import GLiNER2
from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# 1. Create training examples
examples = [
    InputExample(
        text="John works at Google in California.",
        entities={"person": ["John"], "company": ["Google"], "location": ["California"]}
    ),
    InputExample(
        text="Apple released iPhone 15.",
        entities={"company": ["Apple"], "product": ["iPhone 15"]}
    ),
]

# 2. Initialize model and config
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./output",
    num_epochs=10,
    batch_size=8,
    encoder_lr=1e-5,
    task_lr=5e-4
)

# 3. Train
trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
```

### Quick Start with JSONL

```python
# Create train.jsonl file with format:
# {"input": "text here", "output": {"entities": {"type": ["mention1", "mention2"]}}}

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(output_dir="./output", num_epochs=10)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data="train.jsonl")
```

---

## Data Format Overview

GLiNER2 supports **multiple input formats** for maximum flexibility:

### 1. JSONL Files (Recommended for Large Datasets)

Single or multiple JSONL files where each line is a JSON record:

```json
{"input": "John works at Google.", "output": {"entities": {"person": ["John"], "company": ["Google"]}}}
{"input": "Apple released iPhone.", "output": {"entities": {"company": ["Apple"], "product": ["iPhone"]}}}
```

**Usage:**
```python
# Single file
trainer.train(train_data="train.jsonl")

# Multiple files
trainer.train(train_data=["train1.jsonl", "train2.jsonl", "train3.jsonl"])
```

### 2. InputExample List (Recommended for Programmatic Creation)

```python
from gliner2.training.data import InputExample

examples = [
    InputExample(
        text="John works at Google.",
        entities={"person": ["John"], "company": ["Google"]}
    ),
    InputExample(
        text="Apple released iPhone.",
        entities={"company": ["Apple"], "product": ["iPhone"]}
    )
]

trainer.train(train_data=examples)
```

### 3. TrainingDataset Object

```python
from gliner2.training.data import TrainingDataset

# Load from JSONL
dataset = TrainingDataset.load("train.jsonl")

# Or create from examples
dataset = TrainingDataset(examples)

# Add validation and statistics
dataset.validate()
dataset.print_stats()

trainer.train(train_data=dataset)
```

### 4. Raw Dict List

```python
raw_data = [
    {"input": "text", "output": {"entities": {"person": ["John"]}}},
    {"input": "more text", "output": {"entities": {"company": ["Google"]}}}
]

trainer.train(train_data=raw_data)
```

---

## Creating Training Examples

GLiNER2 supports **four types of tasks**:

1. **Entity Extraction** (NER)
2. **Classification** (text classification, sentiment, etc.)
3. **Structured Data Extraction** (JSON-like structures)
4. **Relation Extraction** (relations between entities)

### 1. Entity Extraction

Extract named entities from text.

```python
from gliner2.training.data import InputExample

# Simple entity extraction
example = InputExample(
    text="John Smith works at Google in San Francisco.",
    entities={
        "person": ["John Smith"],
        "company": ["Google"],
        "location": ["San Francisco"]
    }
)

# With entity descriptions (improves model understanding)
example = InputExample(
    text="BERT was developed by Google AI.",
    entities={
        "model": ["BERT"],
        "organization": ["Google AI"]
    },
    entity_descriptions={
        "model": "Machine learning model or architecture",
        "organization": "Company, research lab, or institution"
    }
)
```

**Convenience function:**
```python
from gliner2.training.data import create_entity_example

example = create_entity_example(
    text="John works at Google.",
    entities={"person": ["John"], "company": ["Google"]},
    descriptions={"person": "Human name", "company": "Business organization"}
)
```

### 2. Classification

Classify text into predefined categories.

```python
from gliner2.training.data import InputExample, Classification

# Single-label classification
example = InputExample(
    text="This movie is amazing! Best film of the year.",
    classifications=[
        Classification(
            task="sentiment",
            labels=["positive", "negative", "neutral"],
            true_label="positive"
        )
    ]
)

# Multi-label classification
example = InputExample(
    text="The Python programming tutorial covers machine learning and web development.",
    classifications=[
        Classification(
            task="topic",
            labels=["programming", "machine_learning", "web_dev", "data_science", "mobile"],
            true_label=["programming", "machine_learning", "web_dev"],
            multi_label=True
        )
    ]
)

# With label descriptions
example = InputExample(
    text="The patient shows symptoms of fever and cough.",
    classifications=[
        Classification(
            task="severity",
            labels=["mild", "moderate", "severe"],
            true_label="moderate",
            label_descriptions={
                "mild": "Minor symptoms, no hospitalization needed",
                "moderate": "Some symptoms requiring monitoring",
                "severe": "Life-threatening, requires immediate care"
            }
        )
    ]
)
```

**Convenience function:**
```python
from gliner2.training.data import create_classification_example

example = create_classification_example(
    text="This product is terrible!",
    task="sentiment",
    labels=["positive", "negative", "neutral"],
    true_label="negative"
)
```

### 3. Structured Data Extraction

Extract structured information as JSON-like objects.

```python
from gliner2.training.data import InputExample, Structure, ChoiceField

# Simple structure
example = InputExample(
    text="iPhone 15 Pro costs $999 and comes in titanium color.",
    structures=[
        Structure(
            "product",
            name="iPhone 15 Pro",
            price="$999",
            color="titanium"
        )
    ]
)

# Multiple structures
example = InputExample(
    text="Contact: John Smith, email: john@example.com, phone: 555-1234. "
         "Address: 123 Main St, San Francisco, CA",
    structures=[
        Structure(
            "contact",
            name="John Smith",
            email="john@example.com",
            phone="555-1234"
        ),
        Structure(
            "address",
            street="123 Main St",
            city="San Francisco",
            state="CA"
        )
    ]
)

# With choice fields (classification within structure)
example = InputExample(
    text="Order #12345 for laptop shipped on 2024-01-15.",
    structures=[
        Structure(
            "order",
            order_id="12345",
            product="laptop",
            date="2024-01-15",
            status=ChoiceField(
                value="shipped",
                choices=["pending", "processing", "shipped", "delivered", "cancelled"]
            )
        )
    ]
)

# With field descriptions
example = InputExample(
    text="MacBook Pro M3 available for $1999",
    structures=[
        Structure(
            "product",
            _descriptions={
                "name": "Full product name",
                "price": "Price in USD with $ symbol",
                "processor": "CPU/chip model"
            },
            name="MacBook Pro M3",
            price="$1999",
            processor="M3"
        )
    ]
)
```

**Convenience function:**
```python
from gliner2.training.data import create_structure_example

example = create_structure_example(
    text="iPhone 15 costs $799",
    structure_name="product",
    name="iPhone 15",
    price="$799"
)
```

### 4. Relation Extraction

Extract relationships between entities.

```python
from gliner2.training.data import InputExample, Relation

# Binary relation (head-tail)
example = InputExample(
    text="Elon Musk founded SpaceX in 2002.",
    relations=[
        Relation("founded", head="Elon Musk", tail="SpaceX"),
        Relation("founded_in", head="SpaceX", tail="2002")
    ]
)

# Custom fields
example = InputExample(
    text="The study found that exercise improves mental health.",
    relations=[
        Relation(
            "causal_relation",
            cause="exercise",
            effect="mental health",
            direction="positive"
        )
    ]
)

# Multiple relations
example = InputExample(
    text="John works for Google as an engineer in California.",
    relations=[
        Relation("works_for", head="John", tail="Google"),
        Relation("has_role", head="John", tail="engineer"),
        Relation("located_in", head="Google", tail="California")
    ]
)
```

**Convenience function:**
```python
from gliner2.training.data import create_relation_example

example = create_relation_example(
    text="Steve Jobs co-founded Apple.",
    relation_name="co-founded",
    head="Steve Jobs",
    tail="Apple"
)
```

### 5. Multi-Task Examples

Combine multiple task types in a single example:

```python
example = InputExample(
    text="John Smith works at Google in California. The company is doing well.",
    entities={
        "person": ["John Smith"],
        "company": ["Google"],
        "location": ["California"]
    },
    classifications=[
        Classification(
            task="sentiment",
            labels=["positive", "negative", "neutral"],
            true_label="positive"
        )
    ],
    relations=[
        Relation("works_at", head="John Smith", tail="Google"),
        Relation("located_in", head="Google", tail="California")
    ]
)
```

---

## Data Validation

### Automatic Validation

Enable validation during dataset creation:

```python
from gliner2.training.data import TrainingDataset

dataset = TrainingDataset(examples)

# Validate all examples
try:
    report = dataset.validate(strict=True, raise_on_error=True)
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Shows first 10 errors automatically
```

### Validation Options

```python
# Strict validation: checks if entity mentions exist in text
dataset.validate(strict=True)

# Loose validation: only checks data structure
dataset.validate(strict=False)

# Get report without raising errors
report = dataset.validate(raise_on_error=False)
print(f"Valid: {report['valid']}, Invalid: {report['invalid']}")
print(f"Errors: {report['errors']}")
```

### Individual Example Validation

```python
example = InputExample(
    text="John works at Google",
    entities={"person": ["John"], "company": ["Microsoft"]}  # Error: Microsoft not in text
)

errors = example.validate(strict=True)
if errors:
    for error in errors:
        print(f"Error: {error}")

# Quick check
if example.is_valid(strict=True):
    print("Example is valid!")
```

### Dataset Statistics

```python
dataset = TrainingDataset.load("train.jsonl")

# Print formatted statistics
dataset.print_stats()

# Get statistics as dict
stats = dataset.stats()
print(f"Total examples: {stats['total_examples']}")
print(f"Entity types: {stats['entity_types']}")
print(f"Average text length: {stats['text_length_stats']['mean']}")
```

**Example Output:**
```
============================================================
GLiNER2 Training Dataset Statistics
============================================================
Total examples: 1000

Text lengths: min=15, max=512, mean=123.4

Task Distribution:
  entities_only: 650 (65.0%)
  multi_task: 200 (20.0%)
  classifications_only: 100 (10.0%)
  structures_only: 50 (5.0%)

Entity Types (2450 total mentions):
  person: 850
  company: 600
  location: 500
  product: 500

Classification Tasks:
  sentiment: 300 examples
    - positive: 150
    - negative: 100
    - neutral: 50
============================================================
```

---

## Training Configuration

### TrainingConfig Parameters

```python
from gliner2.training.trainer import TrainingConfig

config = TrainingConfig(
    # Output
    output_dir="./output",           # Where to save checkpoints
    experiment_name="my_experiment",  # Experiment name for logging
    
    # Training steps
    num_epochs=10,                   # Number of epochs
    max_steps=-1,                    # Max steps (-1 = use epochs)
    
    # Batch size
    batch_size=32,                   # Per-device batch size
    eval_batch_size=64,              # Evaluation batch size
    gradient_accumulation_steps=1,   # Gradient accumulation
    
    # Learning rates
    encoder_lr=1e-5,                 # LR for encoder (BERT, etc.)
    task_lr=5e-4,                    # LR for task layers
    
    # Optimizer
    weight_decay=0.01,               # Weight decay
    adam_beta1=0.9,                  # Adam beta1
    adam_beta2=0.999,                # Adam beta2
    adam_epsilon=1e-8,               # Adam epsilon
    max_grad_norm=1.0,               # Gradient clipping
    
    # Learning rate schedule
    scheduler_type="linear",         # "linear", "cosine", "cosine_restarts", "constant"
    warmup_ratio=0.1,                # Warmup ratio (0.1 = 10% of steps)
    warmup_steps=0,                  # Explicit warmup steps (overrides ratio)
    num_cycles=0.5,                  # For cosine_restarts
    
    # Mixed precision
    fp16=True,                       # Use FP16
    bf16=False,                      # Use BF16 (A100/H100)
    
    # Checkpointing
    save_strategy="epoch",           # "epoch", "steps", or "no"
    save_steps=500,                  # Save every N steps (if save_strategy="steps")
    save_total_limit=3,              # Max checkpoints to keep
    save_best=True,                  # Save best model
    metric_for_best="eval_loss",     # Metric for best model
    greater_is_better=False,         # Whether higher = better
    
    # Evaluation
    eval_strategy="epoch",           # "epoch", "steps", or "no"
    eval_steps=500,                  # Evaluate every N steps
    
    # Logging
    logging_steps=50,                # Log every N steps
    logging_first_step=True,         # Log first step
    report_to=["tensorboard"],       # Logging backends: "tensorboard", "wandb"
    
    # Weights & Biases
    wandb_project="gliner2-project", # W&B project name
    wandb_entity=None,               # W&B entity
    wandb_run_name=None,             # W&B run name
    wandb_tags=["ner", "gliner"],    # W&B tags
    wandb_notes="My experiment",     # W&B notes
    
    # Early stopping
    early_stopping=False,            # Enable early stopping
    early_stopping_patience=3,       # Patience (epochs)
    early_stopping_threshold=0.0,    # Min improvement threshold
    
    # DataLoader
    num_workers=4,                   # DataLoader workers
    pin_memory=True,                 # Pin memory for GPU
    prefetch_factor=2,               # Prefetch batches
    
    # Other
    seed=42,                         # Random seed
    deterministic=False,             # Deterministic training (slower)
    gradient_checkpointing=False,    # Use gradient checkpointing (saves memory)
    max_train_samples=-1,            # Limit training samples (-1 = all)
    max_eval_samples=-1,             # Limit eval samples (-1 = all)
    
    # Validation
    validate_data=True,              # Validate data before training
    strict_validation=False,         # Strict entity span checking
)
```

### Common Configurations

**Fast prototyping:**
```python
config = TrainingConfig(
    output_dir="./quick_test",
    num_epochs=3,
    batch_size=16,
    encoder_lr=1e-5,
    task_lr=5e-4,
    max_train_samples=100,
    eval_strategy="no"
)
```

**Production training:**
```python
config = TrainingConfig(
    output_dir="./production_model",
    num_epochs=20,
    batch_size=32,
    gradient_accumulation_steps=2,  # Effective batch size = 64
    encoder_lr=5e-6,
    task_lr=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    scheduler_type="cosine",
    fp16=True,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    save_best=True,
    eval_strategy="steps",
    eval_steps=500,
    early_stopping=True,
    early_stopping_patience=5,
    report_to=["tensorboard", "wandb"],
    wandb_project="gliner2-production"
)
```

**Large model (memory optimization):**
```python
config = TrainingConfig(
    output_dir="./large_model",
    num_epochs=10,
    batch_size=8,                    # Smaller batch
    gradient_accumulation_steps=8,   # Effective batch = 64
    gradient_checkpointing=True,     # Save memory
    fp16=True,
    encoder_lr=1e-6,                 # Lower LR for stability
    task_lr=5e-5,
    max_grad_norm=0.5,               # More aggressive clipping
    num_workers=2                    # Fewer workers
)
```

---

## Running Training

### Method 1: GLiNER2Trainer (Recommended)

Full control with all features:

```python
from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Load model
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Create config
config = TrainingConfig(
    output_dir="./output",
    num_epochs=10,
    batch_size=16,
    encoder_lr=1e-5,
    task_lr=5e-4
)

# Create trainer
trainer = GLiNER2Trainer(
    model=model,
    config=config,
    processor=model.processor,  # Optional, uses model.processor by default
    compute_metrics=None        # Optional custom metrics function
)

# Train
results = trainer.train(
    train_data="train.jsonl",
    eval_data="eval.jsonl"
)

print(f"Training completed in {results['total_time_seconds']:.2f}s")
print(f"Best metric: {results['best_metric']:.4f}")
```

### Method 2: Convenience Function

Quick one-liner for simple cases:

```python
from gliner2.training.trainer import train_gliner2

results = train_gliner2(
    model_path="fastino/gliner2-base-v1",
    train_data="train.jsonl",
    output_dir="./output",
    eval_data="eval.jsonl",
    num_epochs=10,
    batch_size=16,
    encoder_lr=1e-5,
    task_lr=5e-4
)
```

### Training with Custom Metrics

```python
def compute_metrics(model, eval_dataset):
    """Custom metric computation function."""
    # Your custom evaluation logic
    # For example, compute F1 score on entities
    
    metrics = {}
    # ... compute metrics ...
    metrics["f1"] = 0.85
    metrics["precision"] = 0.87
    metrics["recall"] = 0.83
    
    return metrics

trainer = GLiNER2Trainer(
    model=model,
    config=config,
    compute_metrics=compute_metrics
)

trainer.train(train_data=examples, eval_data=eval_examples)
```

### Resume from Checkpoint

```python
trainer = GLiNER2Trainer(model, config)

# Resume training
trainer.resume_from_checkpoint("./output/checkpoints/checkpoint-1000")
trainer.train(train_data=examples)
```

### Distributed Training

```python
# Launch with torchrun or accelerate
# torchrun --nproc_per_node=4 train_script.py

config = TrainingConfig(
    output_dir="./output",
    num_epochs=10,
    local_rank=int(os.environ.get("LOCAL_RANK", -1))  # Auto-detect DDP
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
```

---

## Advanced Topics

### 1. Data Splitting

```python
from gliner2.training.data import TrainingDataset

# Load full dataset
dataset = TrainingDataset.load("full_data.jsonl")

# Split into train/val/test
train_data, val_data, test_data = dataset.split(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle=True,
    seed=42
)

# Save splits
train_data.save("train.jsonl")
val_data.save("val.jsonl")
test_data.save("test.jsonl")

# Use in training
trainer.train(train_data=train_data, eval_data=val_data)
```

### 2. Data Filtering and Sampling

```python
dataset = TrainingDataset.load("train.jsonl")

# Filter by predicate
entity_only = dataset.filter(lambda ex: len(ex.entities) > 0 and not ex.classifications)

# Random sample
small_dataset = dataset.sample(n=100, seed=42)

# Use filtered data
trainer.train(train_data=entity_only)
```

### 3. Combining Multiple Datasets

```python
# Load multiple datasets
dataset1 = TrainingDataset.load("dataset1.jsonl")
dataset2 = TrainingDataset.load("dataset2.jsonl")

# Combine
combined = TrainingDataset()
combined.add_many(dataset1.examples)
combined.add_many(dataset2.examples)

# Or directly from files
trainer.train(train_data=["dataset1.jsonl", "dataset2.jsonl"])
```

### 4. Custom Data Augmentation

```python
from gliner2.training.data import InputExample

def augment_example(example: InputExample) -> List[InputExample]:
    """Create augmented versions of an example."""
    augmented = [example]  # Original
    
    # Shuffle entity order
    if len(example.entities) > 1:
        shuffled_entities = dict(sorted(example.entities.items(), reverse=True))
        augmented.append(InputExample(
            text=example.text,
            entities=shuffled_entities
        ))
    
    return augmented

# Apply augmentation
dataset = TrainingDataset.load("train.jsonl")
augmented_examples = []
for ex in dataset:
    augmented_examples.extend(augment_example(ex))

augmented_dataset = TrainingDataset(augmented_examples)
trainer.train(train_data=augmented_dataset)
```

### 5. Handling Large Datasets

```python
# Use max_samples for testing
config = TrainingConfig(
    output_dir="./output",
    max_train_samples=1000,  # Only use first 1000 samples
    max_eval_samples=200     # Only use first 200 eval samples
)

# Or split large file
dataset = TrainingDataset.load("huge_dataset.jsonl")
chunks = [
    dataset.examples[i:i+10000] 
    for i in range(0, len(dataset), 10000)
]
for i, chunk in enumerate(chunks):
    TrainingDataset(chunk).save(f"chunk_{i}.jsonl")
```

### 6. Learning Rate Schedules

```python
# Linear warmup + linear decay (default)
config = TrainingConfig(
    scheduler_type="linear",
    warmup_ratio=0.1
)

# Cosine annealing
config = TrainingConfig(
    scheduler_type="cosine",
    warmup_ratio=0.05
)

# Cosine with restarts
config = TrainingConfig(
    scheduler_type="cosine_restarts",
    warmup_ratio=0.05,
    num_cycles=3
)

# Constant LR after warmup
config = TrainingConfig(
    scheduler_type="constant",
    warmup_steps=500
)
```

### 7. Memory Optimization Techniques

```python
# Gradient accumulation (reduce batch size, keep effective batch size)
config = TrainingConfig(
    batch_size=4,                    # Small batch fits in memory
    gradient_accumulation_steps=16,  # Effective batch = 64
)

# Gradient checkpointing (trade compute for memory)
config = TrainingConfig(
    gradient_checkpointing=True,
    batch_size=16
)

# Mixed precision training
config = TrainingConfig(
    fp16=True,      # For V100, T4, etc.
    # bf16=True,    # For A100, H100 (better precision)
)

# Reduce workers
config = TrainingConfig(
    num_workers=2,
    prefetch_factor=1
)
```

### 8. Weights & Biases Integration

```python
config = TrainingConfig(
    output_dir="./output",
    report_to=["wandb"],
    wandb_project="my-gliner-project",
    wandb_entity="my-team",
    wandb_run_name="experiment-1",
    wandb_tags=["ner", "entity-extraction", "v1"],
    wandb_notes="Testing new architecture"
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
# Metrics automatically logged to W&B
```

---

## Examples

### Example 1: Named Entity Recognition

```python
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, TrainingDataset
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

# Create NER examples
examples = [
    InputExample(
        text="Tim Cook is the CEO of Apple Inc., based in Cupertino.",
        entities={
            "person": ["Tim Cook"],
            "company": ["Apple Inc."],
            "location": ["Cupertino"]
        }
    ),
    InputExample(
        text="OpenAI released GPT-4 in March 2023.",
        entities={
            "company": ["OpenAI"],
            "model": ["GPT-4"],
            "date": ["March 2023"]
        }
    ),
    # ... more examples
]

# Create and validate dataset
dataset = TrainingDataset(examples)
dataset.validate()
dataset.print_stats()
dataset.save("ner_train.jsonl")

# Train
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./ner_model",
    num_epochs=15,
    batch_size=16,
    encoder_lr=1e-5,
    task_lr=5e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_best=True
)

trainer = GLiNER2Trainer(model, config)
results = trainer.train(train_data="ner_train.jsonl")
```

### Example 2: Sentiment Classification

```python
from gliner2.training.data import InputExample, Classification

examples = [
    InputExample(
        text="This product is amazing! Best purchase ever.",
        classifications=[
            Classification(
                task="sentiment",
                labels=["positive", "negative", "neutral"],
                true_label="positive"
            )
        ]
    ),
    InputExample(
        text="Terrible quality, broke after one day.",
        classifications=[
            Classification(
                task="sentiment",
                labels=["positive", "negative", "neutral"],
                true_label="negative"
            )
        ]
    ),
    # ... more examples
]

# Train
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(output_dir="./sentiment_model", num_epochs=10)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
```

### Example 3: Structured Data Extraction

```python
from gliner2.training.data import InputExample, Structure

examples = [
    InputExample(
        text="Product: iPhone 15 Pro, Price: $999, Color: titanium, Storage: 256GB",
        structures=[
            Structure(
                "product",
                name="iPhone 15 Pro",
                price="$999",
                color="titanium",
                storage="256GB"
            )
        ]
    ),
    InputExample(
        text="Contact John Doe at john@example.com or call 555-0123",
        structures=[
            Structure(
                "contact",
                name="John Doe",
                email="john@example.com",
                phone="555-0123"
            )
        ]
    ),
    # ... more examples
]

# Train
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(output_dir="./structure_model", num_epochs=12)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
```

### Example 4: Multi-Task Learning

```python
examples = [
    InputExample(
        text="John Smith works at Google in California. The company is thriving.",
        entities={
            "person": ["John Smith"],
            "company": ["Google"],
            "location": ["California"]
        },
        classifications=[
            Classification(
                task="sentiment",
                labels=["positive", "negative", "neutral"],
                true_label="positive"
            )
        ],
        relations=[
            {"works_at": {"head": "John Smith", "tail": "Google"}},
            {"located_in": {"head": "Google", "tail": "California"}}
        ]
    ),
    # ... more examples
]

# Train multi-task model
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./multitask_model",
    num_epochs=15,
    batch_size=16
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=examples)
```

### Example 5: Domain-Specific Fine-tuning

```python
# Medical NER example
medical_examples = [
    InputExample(
        text="Patient presented with hypertension and type 2 diabetes.",
        entities={
            "condition": ["hypertension", "type 2 diabetes"]
        },
        entity_descriptions={
            "condition": "Medical condition, disease, or symptom"
        }
    ),
    InputExample(
        text="Prescribed metformin 500mg twice daily.",
        entities={
            "medication": ["metformin"],
            "dosage": ["500mg"],
            "frequency": ["twice daily"]
        }
    ),
    # ... more medical examples
]

# Fine-tune on medical domain
model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
config = TrainingConfig(
    output_dir="./medical_ner",
    num_epochs=20,
    batch_size=16,
    encoder_lr=5e-6,  # Lower LR for fine-tuning
    task_lr=1e-4
)

trainer = GLiNER2Trainer(model, config)
trainer.train(train_data=medical_examples)
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions:**
```python
# 1. Reduce batch size
config = TrainingConfig(batch_size=4)

# 2. Use gradient accumulation
config = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=8  # Effective batch = 32
)

# 3. Enable gradient checkpointing
config = TrainingConfig(gradient_checkpointing=True)

# 4. Use mixed precision
config = TrainingConfig(fp16=True)

# 5. Reduce workers
config = TrainingConfig(num_workers=2)
```

### Issue: Training is Slow

**Solutions:**
```python
# 1. Increase batch size (if memory allows)
config = TrainingConfig(batch_size=64)

# 2. Increase workers
config = TrainingConfig(num_workers=8)

# 3. Use mixed precision
config = TrainingConfig(fp16=True)

# 4. Reduce validation frequency
config = TrainingConfig(
    eval_strategy="steps",
    eval_steps=1000  # Less frequent evaluation
)

# 5. Limit training samples for testing
config = TrainingConfig(max_train_samples=1000)
```

### Issue: Validation Errors

```python
# Check specific errors
dataset = TrainingDataset(examples)
report = dataset.validate(raise_on_error=False)

print(f"Invalid examples: {report['invalid_indices']}")
for error in report['errors'][:10]:
    print(error)

# Fix common issues:
# 1. Entity not in text
example = InputExample(
    text="John works here",
    entities={"person": ["John Smith"]}  # ERROR: "John Smith" not in text
)
# Fix: Use exact match
example = InputExample(
    text="John works here",
    entities={"person": ["John"]}  # OK
)

# 2. Empty entities
example = InputExample(
    text="Some text",
    entities={"person": []}  # ERROR: empty list
)
# Fix: Remove empty entity types
example = InputExample(
    text="Some text",
    entities={}  # OK if other tasks present
)

# 3. Use loose validation during development
dataset.validate(strict=False, raise_on_error=False)
```

### Issue: Model Not Learning

**Solutions:**
```python
# 1. Check learning rates
config = TrainingConfig(
    encoder_lr=1e-5,  # Try different values: 5e-6, 1e-5, 5e-5
    task_lr=5e-4      # Try different values: 1e-4, 5e-4, 1e-3
)

# 2. Increase training epochs
config = TrainingConfig(num_epochs=20)

# 3. Check warmup
config = TrainingConfig(warmup_ratio=0.1)

# 4. Reduce weight decay
config = TrainingConfig(weight_decay=0.001)

# 5. Try different scheduler
config = TrainingConfig(scheduler_type="cosine")

# 6. Check data quality
dataset.print_stats()
dataset.validate()
```

### Issue: Checkpoint Not Saving

```python
# Ensure save strategy is set
config = TrainingConfig(
    save_strategy="epoch",  # or "steps"
    save_steps=500,         # if using "steps"
    save_total_limit=5      # keep last 5 checkpoints
)

# Check output directory permissions
config = TrainingConfig(output_dir="./output")

# Ensure is_main_process for distributed training
# (automatically handled by trainer)
```

### Issue: W&B Not Logging

```python
# 1. Install wandb
# pip install wandb

# 2. Login
# wandb login

# 3. Configure properly
config = TrainingConfig(
    report_to=["wandb"],
    wandb_project="my-project",
    wandb_entity="my-team"  # Optional
)

# 4. Check environment
import wandb
print(wandb.api.api_key)  # Should show your key
```

---

## Best Practices

1. **Always validate data before training:**
   ```python
   dataset.validate()
   dataset.print_stats()
   ```

2. **Start with small subset for testing:**
   ```python
   config = TrainingConfig(max_train_samples=100)
   ```

3. **Use early stopping for long training:**
   ```python
   config = TrainingConfig(
       early_stopping=True,
       early_stopping_patience=5
   )
   ```

4. **Save intermediate checkpoints:**
   ```python
   config = TrainingConfig(
       save_strategy="steps",
       save_steps=500,
       save_best=True
   )
   ```

5. **Monitor training with tensorboard or W&B:**
   ```python
   config = TrainingConfig(report_to=["tensorboard"])
   # Then: tensorboard --logdir ./output/logs
   ```

6. **Use descriptive entity types and add descriptions:**
   ```python
   example = InputExample(
       text="...",
       entities={...},
       entity_descriptions={
           "person": "Full name of a person",
           "company": "Business organization name"
       }
   )
   ```

7. **Split your data properly:**
   ```python
   train, val, test = dataset.split(0.8, 0.1, 0.1)
   ```

8. **Use appropriate learning rates:**
   - Encoder LR: `1e-6` to `5e-5` (typically `1e-5`)
   - Task LR: `1e-4` to `1e-3` (typically `5e-4`)

9. **Enable validation during training:**
   ```python
   config = TrainingConfig(
       eval_strategy="epoch",
       save_best=True
   )
   ```

10. **Document your experiments:**
    ```python
    config = TrainingConfig(
        experiment_name="v1_medical_ner",
        wandb_notes="Testing with augmented data"
    )
    ```

---

## Additional Resources

- [GLiNER2 GitHub Repository](https://github.com/fastino/GLiNER2)
- [Model Inference Tutorial](./INFERENCE.md)
- [API Reference](./API_REFERENCE.md)
- [Examples Directory](../examples/)

---

## Summary

GLiNER2 provides a flexible and powerful framework for training information extraction models:

- **Multiple data formats**: JSONL, InputExample, TrainingDataset, raw dicts
- **Four task types**: Entities, Classifications, Structures, Relations
- **Comprehensive validation**: Automatic data validation and statistics
- **Production-ready training**: FP16, gradient accumulation, distributed training
- **Extensive configuration**: 40+ config options for fine-grained control
- **Easy to use**: Quick start in 10 lines of code

Start with the Quick Start examples and gradually explore advanced features as needed!

