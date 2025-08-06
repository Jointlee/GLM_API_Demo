from datetime import datetime
import os
from utils import load_json_file, load_jsonl_file, save_json_file
from glm_api import GLMAPI
from data_process import genrate_segment_dataset, generate_rhetoric_dataset


def process_inference_results(work_dir, test_data_path, predictions_source: list|str):
    """
    通用的推理结果处理函数
    
    Args:
        work_dir: 工作目录
        test_data_path: 测试数据路径
        predictions_source: 预测结果源（jsonl文件路径或直接的预测结果列表）
    """
    all_test_data = load_json_file(test_data_path)
    
    if isinstance(predictions_source, str):
        assert predictions_source.endswith('.jsonl'), "predictions_source must be a jsonl file path"
        jsonl_data = load_jsonl_file(predictions_source)
        predictions = []
        for item in jsonl_data:
            if 'predict' in item:
                pred = item['predict']
                # 判断predict格式并提取
                if isinstance(pred, list) and len(pred) > 0:
                    predictions.append(pred[0])  
                elif isinstance(pred, str):
                    predictions.append(pred)
                else:
                    predictions.append(str(pred))
            else:
                predictions.append("")
    elif isinstance(predictions_source, list):
        predictions = predictions_source
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
    
    # 验证长度匹配
    assert len(predictions) == len(all_test_data), f"Length mismatch: {len(predictions)} vs {len(all_test_data)}"
    
    for pred, record in zip(predictions, all_test_data):
        record["raw_data"]["llm_pred"] = pred.strip() if isinstance(pred, str) else str(pred).strip()

    results_file = os.path.join(work_dir, "results.json")
    save_json_file(data=all_test_data, path=results_file)
    
    return results_file


def api_infer(model="glm-4-plus", test_data_path=None, task_description="修辞检测"):
    """使用GLM API进行批处理推理"""
    
    assert test_data_path, "test_data_path must be provided for API inference"
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    work_dir = f"saves/api_infer_{timestamp}"
    os.makedirs(work_dir, exist_ok=True)
    
    # 初始化API客户端
    api_key = "your_api_key_here" 
    api_client = GLMAPI(api_key=api_key, model=model)

    test_data = load_json_file(test_data_path)
    
    # 进行批处理推理
    # print("开始API批处理推理...")
    # results = api_client.batch_process(
    #     infer_data=load_json_file(test_data_path),
    #     description=task_description
    # )
    print("API异步推理")
    results:list = api_client.async_process(
        infer_data=test_data,
    )
    process_inference_results(work_dir, test_data_path, results)
    return work_dir


if __name__ == "__main__":
    
    _, test_data_path = genrate_segment_dataset()
    # 示例：使用GLM API进行批处理推理

    result_path = api_infer(model="glm-4.5", test_data_path=test_data_path, task_description="测试分词")