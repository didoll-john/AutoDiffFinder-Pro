import cv2
import numpy as np
import gradio as gr

def find_similar_regions(image: np.ndarray, min_height: int = 100) -> np.ndarray:
    """
    在图像中寻找两个高度相似的区域并将图像分割
    
    Args:
        image: 输入图像
        min_height: 最小区域高度
    
    Returns:
        差异图像
    """
    # 1. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 创建SIFT检测器并检测特征点
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # 3. 对特征点按y坐标排序
    keypoints_sorted = sorted(keypoints, key=lambda kp: kp.pt[1])
    
    # 获取原始图片的高度和宽度
    height, width = image.shape[:2]
    best_match_count = 0
    best_matches = []
    best_top_points = []
    best_bottom_points = []
    
    # 设定step为len(keypoints_sorted)//200, 如果len(keypoints_sorted)小于200，则step为1
    step = max(1, len(keypoints_sorted)//200)
    # 遍历图像的不同分割点，寻找最佳匹配位置
    for split_y in range(min_height, height - min_height, step):
        # 根据分割点将特征点分为上下两部分
        top_points = [kp for kp in keypoints_sorted if kp.pt[1] < split_y]
        bottom_points = [kp for kp in keypoints_sorted if kp.pt[1] >= split_y]
        
        # 确保上下两部分都有特征点
        if len(top_points) > 0 and len(bottom_points) > 0:
            # 获取上下两部分特征点对应的描述符
            top_desc = np.array([descriptors[keypoints.index(kp)] for kp in top_points])
            bottom_desc = np.array([descriptors[keypoints.index(kp)] for kp in bottom_points])
            
            # 使用BFMatcher替代FLANN，更适合SIFT特征匹配
            bf = cv2.BFMatcher()
            # 对每个特征点找到最佳的2个匹配点
            matches = bf.knnMatch(top_desc, bottom_desc, k=2)
            
            # 使用比率测试筛选出好的匹配点
            good_matches = []
            for match in matches:
                # 确保找到了两个匹配点
                if len(match) == 2:
                    m, n = match
                    # 如果最佳匹配的距离小于次佳匹配的0.7倍，则认为是好的匹配
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # 如果当前分割点的匹配数量更多，则更新最佳匹配结果
            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_matches = good_matches
                best_top_points = top_points
                best_bottom_points = bottom_points
    
    # 计算best_matchs中所有匹配点对在y轴的距离，并找出其中出现最频繁的距离
    distances = [abs(best_top_points[m.queryIdx].pt[1] - best_bottom_points[m.trainIdx].pt[1]) for m in best_matches]
    # 将距离值四舍五入到最接近的整数
    distances = [round(d) for d in distances]
    distance_counts = {}
    for distance in distances:
        distance_counts[distance] = distance_counts.get(distance, 0) + 1
    most_frequent_distance = max(distance_counts, key=distance_counts.get)
    
    # 创建新图像
    new_image = np.vstack((
        image[most_frequent_distance:height, 0:width],  # 裁剪后的图片
        np.full((most_frequent_distance, width, 3), 255, dtype=np.uint8)  # 白色填充
    ))

    diff = cv2.absdiff(image, new_image)
    enhanced_diff = cv2.multiply(diff, 3.5)
    
    return enhanced_diff  # 只返回差异图像

def process_image(input_image):
    """
    处理上传的图像并返回结果
    
    Args:
        input_image: 输入图像路径或numpy数组
    
    Returns:
        差异图像
    """
    if input_image is None:
        return None
    
    if isinstance(input_image, str):
        image = cv2.imread(input_image)
    else:
        image = input_image
        
    if image is None:
        return None
    
    # 确保图像是BGR格式
    if len(image.shape) == 2:  # 如果是灰度图
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    diff_image = find_similar_regions(image)
    return diff_image

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="图像相似区域检测器") as interface:
        gr.Markdown("## 图像相似区域检测器\n上传一张图片，自动检测并显示相似区域。")
        
        with gr.Row():
            input_image = gr.Image(label="输入图像", type="numpy")
            diff_output = gr.Image(label="差异图像")
            
        process_btn = gr.Button("开始处理")
        process_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[diff_output]
        )

        # 添加示例图片
        gr.Examples(
            examples=["Examples/example.jpg"],
            inputs=input_image,
        )
    
    return interface

def main():
    interface = create_interface()
    interface.launch(share=False)

if __name__ == "__main__":
    main()