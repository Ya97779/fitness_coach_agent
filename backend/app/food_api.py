"""食物热量API模块 - 使用天行数据API查询食物营养信息"""

import os
import json
import http.client
import urllib
from dotenv import load_dotenv

load_dotenv()

# 天行数据API配置
API_KEY = os.getenv("TianxingFood_API_KEY")
API_HOST = "apis.tianapi.com"
API_PATH = "/nutrient/index"

# 备用本地数据（当API不可用时使用）
FALLBACK_DATA = {
    "苹果": {"calories": 52, "protein": 0.3, "fat": 0.2, "carbs": 14},
    "香蕉": {"calories": 91, "protein": 1.1, "fat": 0.3, "carbs": 23},
    "米饭": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28},
    "鸡蛋": {"calories": 78, "protein": 6.3, "fat": 5.3, "carbs": 0.6},
    "鸡胸肉": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0},
    "兰州拉面": {"calories": 500, "protein": 15, "fat": 15, "carbs": 70},
    "可乐": {"calories": 150, "protein": 0, "fat": 0, "carbs": 39},
    "油条": {"calories": 385, "protein": 6, "fat": 17, "carbs": 51}
}


def search_food_nutrient(food_name: str) -> dict:
    """调用天行数据API查询食物营养信息
    
    Args:
        food_name: 食物名称
        
    Returns:
        包含热量、蛋白质、脂肪、碳水化合物的字典，或None表示未找到
    """
    if not API_KEY:
        print("未配置TianxingFood_API_KEY，使用备用数据")
        return FALLBACK_DATA.get(food_name, None)
    
    try:
        # 构建请求参数
        params = urllib.parse.urlencode({
            'key': API_KEY,
            'mode': '0',
            'word': food_name
        })
        
        # 发送请求
        conn = http.client.HTTPSConnection(API_HOST)
        headers = {'Content-type': 'application/x-www-form-urlencoded'}
        conn.request('POST', API_PATH, params, headers)
        
        # 获取响应
        response = conn.getresponse()
        result = response.read()
        conn.close()
        
        # 解析响应
        data = json.loads(result.decode('utf-8'))
        
        # 调试：打印完整响应
        print(f"API响应数据: {json.dumps(data, ensure_ascii=False)}")
        
        # 检查API返回码
        code = data.get("code", 0)
        
        # 天行数据API成功时code=200
        if code == 200:
            # 数据在 result.list 中
            result_data = data.get("result", {})
            food_list = result_data.get("list", [])
            
            if food_list and len(food_list) > 0:
                nutrient = food_list[0]
                
                # API字段映射
                # rl = 热量, dbz = 蛋白质, zf = 脂肪, shhf = 碳水化合物
                calories = nutrient.get("rl", 0)
                protein = nutrient.get("dbz", 0)
                fat = nutrient.get("zf", 0)
                carbs = nutrient.get("shhf", 0)
                
                # 如果获取到有效数据（热量大于0）
                if int(calories) > 0:
                    return {
                        "calories": int(calories),
                        "protein": float(protein),
                        "fat": float(fat),
                        "carbs": float(carbs),
                        "source": "天行数据API"
                    }
        
        # 没有找到数据
        print(f"API未找到'{food_name}'的信息: {data.get('msg', '未知')}")
        return FALLBACK_DATA.get(food_name, None)
    
    except Exception as e:
        print(f"API调用失败: {e}")
        return FALLBACK_DATA.get(food_name, None)


def search_food_calories(food_name: str) -> str:
    """搜索食物热量（用于Agent工具）
    
    如果API和本地数据都查不到，返回None让大模型回答
    """
    result = search_food_nutrient(food_name)
    
    if result:
        return f"食物: {food_name}, 热量: {result['calories']} kcal, 蛋白质: {result['protein']}g, 脂肪: {result['fat']}g, 碳水: {result['carbs']}g (来源: {result.get('source', '本地')})"
    
    # 返回None表示未找到，让大模型直接回答
    return None


def get_food_details(food_name: str) -> str:
    """获取食物详细营养信息（格式化输出）"""
    result = search_food_nutrient(food_name)
    
    if not result:
        return f"未找到'{food_name}'的营养信息"
    
    return (
        f"🍽️ **{food_name}**\n"
        f"数据来源: {result.get('source', '本地数据')}\n"
        f"🔥 热量: {result.get('calories', 0)} kcal/100g\n"
        f"💪 蛋白质: {result.get('protein', 0)} g\n"
        f"🥑 脂肪: {result.get('fat', 0)} g\n"
        f"🍞 碳水: {result.get('carbs', 0)} g"
    )
