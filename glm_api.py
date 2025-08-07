from zai import ZhipuAiClient
import time
import json
import os
import tempfile
import requests
from typing import List, Dict

class GLMAPI:
    def __init__(self, api_key="", model='glm-4-plus'):
        self.client = ZhipuAiClient(api_key=api_key)
        self.api_key = api_key
        self.model = model
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.batch_max_retries = 48  # 最大重试次数，48次，每次等待30分钟，总计24小时
        self.async_max_retries = 120  # 异步任务最大重试次数，60次，每次等待1秒，总计120秒
        self.max_concurrent_tasks = 20  # 最大并发任务数
        
    def _create_batch_file(self, infer_data: List[dict]) -> str:
        """创建batch请求文件"""
        batch_requests = []
        for i, item in enumerate(infer_data):
            system = item.get('system', '')
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            
            # 构建完整的prompt
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{instruction}\n\n{input_text}"}
            ]
            
            batch_request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": messages
                },
                "max_tokens": 1024,
            }
            batch_requests.append(json.dumps(batch_request, ensure_ascii=False))
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
        temp_file.write('\n'.join(batch_requests))
        temp_file.close()
        
        return temp_file.name

    def batch_process(self, infer_data: List[dict] = None, description: str = '') -> List[str]:

        assert self.model != "glm-4.5", "GLM-4.5模型不支持批处理，请使用其他模型"
        # 创建batch文件
        batch_file_path = self._create_batch_file(infer_data)
        
        try:
            # 1.上传Batch文件
            with open(batch_file_path, "rb") as f:
                batchFile = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            print(f"batch文件id: {batchFile.id}")
            
            # 2.创建Batch
            createBatch = self.client.batches.create(
                input_file_id=batchFile.id,
                endpoint="/v4/chat/completions",
                metadata={
                    "description": description if description else "批处理任务"
                }
            )
            print(f"创建Batch成功：{createBatch}")
            
            # 3.轮询Batch处理结果
            results = self._poll_batch_results(createBatch.id, len(infer_data))
            return results
            
        finally:
            # 清理临时文件
            if os.path.exists(batch_file_path):
                os.unlink(batch_file_path)

    def _poll_batch_results(self, batch_id: str, expected_count: int) -> List[str]:
        """轮询batch结果"""
        for attempt in range(self.batch_max_retries):
            try:
                retrieve = self.client.batches.retrieve(batch_id)
                print(f"Batch状态: {retrieve.status}")
                
                if retrieve.status == "completed":
                    # 4.下载batch结果
                    content = self.client.files.content(retrieve.output_file_id)
                    
                    # 创建临时文件保存结果
                    temp_result_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False, encoding='utf-8')
                    content.write_to_file(temp_result_file.name)
                    
                    # 解析结果
                    results = self._parse_batch_results(temp_result_file.name, expected_count)
                    
                    # 清理临时文件
                    os.unlink(temp_result_file.name)
                    
                    return results
                    
                elif retrieve.status == "failed":
                    print(f"Batch处理失败: {retrieve}")
                    return ["Tasks failed"] * expected_count
                    
                else:
                    print(f"等待Batch完成，当前状态: {retrieve.status}")
                    time.sleep(1800)  # 等待30分钟后重试
                    
            except Exception as e:
                print(f"轮询batch结果时出错: {str(e)}")
                time.sleep(secends=1800)  # 等待30分钟后重试
        
        print("Batch处理超时")
        return ["Tasks failed"] * expected_count
    
    def _parse_batch_results(self, result_file_path: str, expected_count: int) -> List[str]:
        """解析batch结果文件"""
        results = [""] * expected_count
        
        try:
            with open(result_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result_item = json.loads(line)
                        custom_id = result_item.get('custom_id', '')
                        if custom_id.startswith('request-'):
                            index = int(custom_id.split('-')[1])
                            if 'response' in result_item and 'body' in result_item['response']:
                                content = result_item['response']['body']['choices'][0]['message']['content']
                                results[index] = content.strip()
                            else:
                                results[index] = "Tasks failed"
        except Exception as e:
            print(f"解析batch结果时出错: {str(e)}")
            return ["Tasks failed"] * expected_count
        
        return results
    
    def async_process(self, infer_data: List[dict]) -> List[str]:
        """异步批处理数据，限制并发数在20以内"""
        total_data = len(infer_data)
        results = [""] * total_data
        total_batches = (total_data + self.max_concurrent_tasks - 1) // self.max_concurrent_tasks

        for batch_idx, batch_start in enumerate(range(0, total_data, self.max_concurrent_tasks), 1):
            batch_end = min(batch_start + self.max_concurrent_tasks, total_data)
            batch_data = infer_data[batch_start:batch_end]
            
            print(f"处理批次 {batch_idx}/{total_batches} (任务 {batch_start+1}-{batch_end}/{total_data})")

            # 提交当前批次的任务
            task_ids = self._submit_async_tasks(batch_data)

            # 轮询当前批次的结果
            batch_results = self._poll_async_tasks(task_ids)

            # 将结果填入总结果列表
            for i, result in enumerate(batch_results):
                results[batch_start + i] = result
                
            print(f"批次 {batch_idx}/{total_batches} 完成")

        print(f"所有任务完成，共处理 {total_data} 个任务")
        return results
    
    def _submit_async_tasks(self, batch_data: List[dict]) -> List[str]:
        """提交异步任务"""
        task_ids = []
        
        for item in batch_data:
            system = item.get('system', '')
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            
            # 构建完整的prompt
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{instruction}\n\n{input_text}"}
            ]
            
            try:
                response = self.client.chat.asyncCompletions.create(
                    model=self.model,
                    messages=messages
                )
                task_ids.append(response.id)
            except Exception as e:
                print(f"提交任务失败: {str(e)}")
                task_ids.append(None)
        
        return task_ids
    
    def _poll_async_tasks(self, task_ids: List[str]) -> List[str]:
        """轮询异步任务结果"""
        results = [""] * len(task_ids)
        
        for attempt in range(self.async_max_retries):
            all_done = True
            
            for idx, task_id in enumerate(task_ids):
                if results[idx] or task_id is None:  # 已完成或任务ID为空
                    continue
                    
                try:
                    resp = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
                    
                    if resp.task_status == "SUCCESS":
                        results[idx] = resp.choices[0].message.content.strip()
                    elif resp.task_status == "FAILED":
                        results[idx] = "Task failed"
                    else:
                        all_done = False
                        
                except Exception as e:
                    print(f"轮询任务 {task_id} 时出错: {str(e)}")
                    results[idx] = f"API Error: {str(e)}"
            
            if all_done:
                break
                
            time.sleep(1)  # 等待1秒后重试
        
        # 处理未完成的任务
        for idx, result in enumerate(results):
            if not result and task_ids[idx] is not None:
                results[idx] = "Task timeout"
        
        return results

    def http_call(self, messages: List[Dict], temperature: float = 0.6, max_tokens: int = 1024) -> str:
        """HTTP方式调用智谱AI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API调用失败: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"HTTP调用出错: {str(e)}")
            return f"HTTP Error: {str(e)}"

    def http_process(self, infer_data: List[dict], temperature: float = 0.6) -> List[str]:
        """使用HTTP方式批量处理数据"""
        results = []
        total_data = len(infer_data)
        
        for i, item in enumerate(infer_data):
            print(f"处理任务 {i+1}/{total_data}")
            
            system = item.get('system', '')
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            
            # 构建完整的prompt
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{instruction}\n\n{input_text}"}
            ]
            
            result = self.http_call(messages, temperature)
            results.append(result)
            
            # 避免请求过于频繁
            time.sleep(0.1)
        
        print(f"HTTP批量处理完成，共处理 {total_data} 个任务")
        return results


if __name__ == "__main__":
    pass
