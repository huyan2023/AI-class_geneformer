from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda"  # the device to load the model onto
model_name = "ctheodoris/Geneformer"
model = AutoModelForCausalLM.from_pretrained(
    "ctheodoris/Geneformer",
    torch_dtype="auto",
    # device_map="auto",
    is_decoder=True
)
tokenizer = AutoTokenizer.from_pretrained("ctheodoris/Geneformer")

# 导入数据
dataset = load_dataset("example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset")
print(dataset)
