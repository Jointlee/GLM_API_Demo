import os
from typing import Dict, List, Any
from utils import load_json_file, save_json_file
from instructions import RHETORIC_INSTRUCTION, RHETORIC_SYSTEM, SEGMENTATION_INSTRUCTION, SEGMENTATION_SYSTEM


def generate_dataset_from_raw(raw_data: List[Dict], system_prompt: str, instruction: str, input_processor: callable, output_processor: callable = None, include_output: bool = True) -> List[Dict]:
    dataset = []
    
    for item in raw_data:
        input_str = input_processor(item)
        
        output_str = ""
        if include_output and output_processor:
            output_str = output_processor(item)

        dataset.append({
            "system": system_prompt,
            "instruction": instruction,
            "input": input_str,
            "output": output_str,
            "history": [],
            "raw_data": item
        })
    
    return dataset


def generate_rhetoric_dataset(dataset_path="rhetoric_data", output_path="rhetoric_data"):
    """修辞识别数据集生成"""
    for fname in ["train.json", "test.json"]:
        assert os.path.exists(os.path.join(dataset_path, fname)), f"缺少{fname}文件"

    os.makedirs(output_path, exist_ok=True)
    
    raw_data_train = load_json_file(os.path.join(dataset_path, "train.json"))
    raw_data_test = load_json_file(os.path.join(dataset_path, "test.json"))

    def process_rhetoric_input(item):
        sentence = item.get('sentence', '')
        source = item.get('source', '')
        source = "" if source == "无" else source
        return f"句子：{sentence}{source}\n判断："
    
    def process_rhetoric_output(item):
        rhetoric_type = item.get('type', '').strip()
        if rhetoric_type == "" or rhetoric_type == "无转义":
            return "否，该句没有使用修辞手法。"
        else:
            return "是，该句含有修辞手法。"
    
    train_dataset = generate_dataset_from_raw(
        raw_data=raw_data_train,
        system_prompt=RHETORIC_SYSTEM,
        instruction=RHETORIC_INSTRUCTION,
        input_processor=process_rhetoric_input,
        output_processor=process_rhetoric_output,
        include_output=True
    )
    test_dataset = generate_dataset_from_raw(
        raw_data=raw_data_test,
        system_prompt=RHETORIC_SYSTEM,
        instruction=RHETORIC_INSTRUCTION,
        input_processor=process_rhetoric_input,
        output_processor=process_rhetoric_output,
        include_output=False
    )
    
    save_json_file(train_dataset, os.path.join(output_path, "train_for_llm.json"))
    save_json_file(test_dataset, os.path.join(output_path, "test_for_llm.json"))

    print(f"修辞识别数据集已生成，共{len(train_dataset)}条训练数据和{len(test_dataset)}条测试数据，保存到: {output_path}")
    return 


def genrate_segment_dataset(dataset_path="segment_data", output_path="segment_data"):
    """分词数据集生成"""
    for fname in ["train.json", "test.json"]:
        assert os.path.exists(os.path.join(dataset_path, fname)), f"缺少{fname}文件"

    os.makedirs(output_path, exist_ok=True)
    
    raw_data_train = load_json_file(os.path.join(dataset_path, "train.json"))
    raw_data_test = load_json_file(os.path.join(dataset_path, "test.json"))

    def process_segment_input(item):
        sentence = item.get('sentence', '')
        return f"输入：{sentence}"
    
    def process_segment_output(item):
        return item.get('segmentation', '').strip() or ""
    
    train_dataset = generate_dataset_from_raw(
        raw_data=raw_data_train,
        system_prompt=SEGMENTATION_SYSTEM,
        instruction=SEGMENTATION_INSTRUCTION,
        input_processor=process_segment_input,
        output_processor=process_segment_output,
        include_output=True
    )
    test_dataset = generate_dataset_from_raw(
        raw_data=raw_data_test,
        system_prompt=SEGMENTATION_SYSTEM,
        instruction=SEGMENTATION_INSTRUCTION,
        input_processor=process_segment_input,
        output_processor=process_segment_output,
        include_output=False
    )
    
    save_json_file(train_dataset, os.path.join(output_path, "train_for_llm.json"))
    save_json_file(test_dataset, os.path.join(output_path, "test_for_llm.json"))
    print(f"分词数据集已生成，共{len(train_dataset)}条训练数据和{len(test_dataset)}条测试数据，保存到: {output_path}")
    return os.path.join(output_path, "train_for_llm.json"), os.path.join(output_path, "test_for_llm.json")

