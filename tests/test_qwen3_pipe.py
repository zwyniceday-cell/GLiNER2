"""
Pipeline test: initialize Qwen3 tokenizer/config and run a tiny training loop.

This is a demo-style test similar to other tests in this repo.
"""

from gliner2.model import Extractor
from gliner2.training.data import InputExample
from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer


def test_qwen3_training_pipeline():
    print("=" * 80)
    print("QWEN3 TRAINING PIPELINE TEST")
    print("=" * 80)

    base_model = "Qwen/Qwen3-0.6B"
    print(f"\nInitializing model from base config: {base_model}")
    model = Extractor.init_from_base_model(
        base_model,
        encoder_overrides={"num_hidden_layers": 2},
        max_width=8,
        counting_layer="count_lstm",
        token_pooling="first",
    )
    print("Model initialized successfully!\n")

    examples = [
        InputExample(
            text="John works at Qwen in Shanghai.",
            entities={"person": ["John"], "company": ["Qwen"], "location": ["Shanghai"]},
        ),
        InputExample(
            text="Alice joined Qwen as a researcher in Beijing.",
            entities={"person": ["Alice"], "company": ["Qwen"], "location": ["Beijing"]},
        ),
    ]

    config = TrainingConfig(
        output_dir="./output_qwen3_pipe",
        experiment_name="qwen3-pipe",
        num_epochs=1,
        max_steps=1,
        batch_size=1,
        eval_strategy="no",
        fp16=False,
        bf16=False,
        num_workers=0,
        pin_memory=False,
        validate_data=True,
    )

    trainer = GLiNER2Trainer(model=model, config=config)
    results = trainer.train(train_data=examples)

    print("\nTraining completed. Summary:")
    print(results)
    print("=" * 80)


if __name__ == "__main__":
    test_qwen3_training_pipeline()
