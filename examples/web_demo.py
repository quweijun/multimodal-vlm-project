"""
Web演示界面（基于Gradio）
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import tempfile
from PIL import Image

from models.qwen_vl_wrapper import QwenVLWrapper
from inference.visual_grounding import VisualGrounding
from inference.spatial_reasoning import SpatialReasoning

class WebDemo:
    """Web演示类"""
    
    def __init__(self):
        print("正在加载模型...")
        self.model = QwenVLWrapper()
        self.grounding = VisualGrounding(self.model)
        self.reasoning = SpatialReasoning(self.model)
        print("模型加载完成！")
    
    def chat_interface(self, image, question, history):
        """聊天界面"""
        if image is None:
            return "请先上传图像", history
        
        # 保存临时图像
        if isinstance(image, str):
            image_path = image
        else:
            # 处理上传的图像
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, "temp_image.jpg")
            image.save(image_path)
        
        try:
            # 获取模型回复
            response = self.model.chat(image_path, question)
            
            # 更新历史
            history.append((question, response))
            
            return "", history
        
        except Exception as e:
            error_msg = f"处理错误: {str(e)}"
            history.append((question, error_msg))
            return "", history
    
    def analyze_image(self, image):
        """分析图像"""
        if image is None:
            return "请先上传图像"
        
        # 保存临时图像
        temp_dir = tempfile.gettempdir()
        image_path = os.path.join(temp_dir, "temp_analyze.jpg")
        image.save(image_path)
        
        try:
            # 获取图像描述
            caption = self.model.image_captioning(image_path)
            
            # 场景布局分析
            layout = self.reasoning.analyze_scene_layout(image_path)
            
            # 深度估计
            depth = self.reasoning.depth_estimation(image_path)
            
            result = f"""## 图像分析结果

### 图像描述:
{caption}

### 场景布局分析:
{layout}

### 深度关系分析:
{depth}
"""
            return result
        
        except Exception as e:
            return f"分析错误: {str(e)}"
    
    def create_demo(self):
        """创建演示界面"""
        with gr.Blocks(title="多模态视觉语言模型演示", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎯 多模态视觉语言模型演示")
            gr.Markdown("基于 Qwen-VL 的视觉问答、图像描述和空间推理演示")
            
            with gr.Tab("💬 对话演示"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="上传图像",
                            type="pil",
                            sources=["upload", "clipboard"]
                        )
                        question_input = gr.Textbox(
                            label="输入问题",
                            placeholder="请输入关于图像的问题...",
                            lines=3
                        )
                        submit_btn = gr.Button("发送", variant="primary")
                        clear_btn = gr.Button("清空")
                    
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="对话历史",
                            height=500
                        )
                
                # 事件绑定
                submit_btn.click(
                    self.chat_interface,
                    inputs=[image_input, question_input, chatbot],
                    outputs=[question_input, chatbot]
                )
                
                clear_btn.click(
                    lambda: (None, "", []),
                    outputs=[image_input, question_input, chatbot]
                )
            
            with gr.Tab("🔍 图像分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        analyze_image_input = gr.Image(
                            label="上传待分析图像",
                            type="pil",
                            sources=["upload", "clipboard"]
                        )
                        analyze_btn = gr.Button("分析图像", variant="primary")
                    
                    with gr.Column(scale=2):
                        analysis_output = gr.Markdown(
                            label="分析结果"
                        )
                
                analyze_btn.click(
                    self.analyze_image,
                    inputs=[analyze_image_input],
                    outputs=[analysis_output]
                )
            
            with gr.Tab("📚 使用说明"):
                gr.Markdown("""
                ## 使用说明
                
                ### 💬 对话演示
                - 上传一张图像
                - 在文本框中输入问题
                - 点击"发送"获取模型回答
                
                ### 🔍 图像分析
                - 上传一张图像
                - 点击"分析图像"获取详细分析
                - 包括图像描述、场景布局和深度关系
                
                ### 支持的问题类型:
                - 图像内容描述
                - 物体识别和定位
                - 空间关系分析
                - 场景理解
                - 常识推理
                
                ### 示例问题:
                - "描述这张图片"
                - "图片中有哪些物体？"
                - "人和车的位置关系是什么？"
                - "这是什么场景？"
                - "图片中可能正在发生什么？"
                """)
        
        return demo

def main():
    """启动Web演示"""
    demo = WebDemo()
    app = demo.create_demo()
    
    # 启动服务
    print("启动Web演示服务...")
    print("访问 http://localhost:7860 查看演示")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()