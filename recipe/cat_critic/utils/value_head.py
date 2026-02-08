def load_categorical_valuehead_model(local_path, torch_dtype, model_config, trust_remote_code):
    from transformers import AutoModelForTokenClassification

    assert hasattr(model_config, "num_labels")
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )

    # assert model.score.out_features > 128
    model.score.weight.data.zero_()
    model.score.bias.data.zero_()
    return model
