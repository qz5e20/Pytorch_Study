import io
import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
import json

# 读取 json 文件: ImageNet的类别名称
with open("idx_class.json") as f:
    idx_class = json.load(f)

device = 'cuda'
model = models.densenet161(pretrained=True) # 预训练模型
model.to(device)
model.eval() # 进行Inference
# 定义模型函数
# def create_model():
#     model_path = "densenet161.pth" # 模型路径
#     model = models.densenet161(pretrained=True) # 下载预训练模型
#     model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
#     model.eval() # 模型验证
#     return model

# 定义transform
def image_transformer(image_data):
    # 变换操作定义
    transform = transforms.Compose([
        transforms.Resize(256), # 改变size=256
        transforms.CenterCrop(224), # 中心裁剪size=224
        transforms.ToTensor(), # 转换为tensor类型
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])
    # 读取图片
    image = Image.open(io.BytesIO(image_data)) # 读取图片, BytesIO实现了在内存中读写bytes
    trans = transform(image).unsqueeze(0)
    #print("trans shape : ", trans.shape)
    return trans

# 定义预测图片的函数
def predict_image(image_data):
    # 读取转换后的图片
    image_tensor = image_transformer(image_data).to(device)
    # 模型预测
    output = model(image_tensor)
    # 获取最大概率值得下标
    #print('output shape : ', output.shape)
    _, prediction = output.max(1)
    #print('prediction : ', prediction.shape)
    # 获取目标索引
    object_index = prediction.item()
    #print("object_index type : ", type(object_index))
    # 返回对应的类别
    return idx_class[object_index]

# 定义预测图片的函数（多张图片）
def batch_prediction(image_batch):
    image_tensors = [image_transformer(img) for img in image_batch] # 对每一张图片进行transform
    tensor = torch.cat(image_tensors).to(device) # 合并所有tensor
    outputs = model(tensor) # 预测输出
    _, predictions = outputs.max(1) # 获取每张图片预测率最大值的下标
    predictions_ids = predictions.tolist() # 结果保存为List
    return [idx_class[ida] for ida in predictions_ids]


from flask import Flask, request, jsonify
from service_streamer import ThreadedStreamer

import ssl
# 在使用URLopen方法的时候，当目标网站使用的是自签名的证书时就会抛出这个错误
# 全局取消证书验证
ssl._create_default_https_context = ssl._create_unverified_context

# 创建Flask app 和 模型
app = Flask(__name__)

# route
@app.route('/predict', methods=['POST'])
def predicted():
    if 'image' not in request.files:
        return jsonify({'error', 'Image not found'}),400
    if request.method == 'POST':
        image = request.files['image'].read() # 读取图片
        object_name = predict_image(image) # 预测结果
        return jsonify({'object_name':object_name}) #  jsonify 确保 response为 json格式


streamer = ThreadedStreamer(batch_prediction, batch_size=64) # 每次预测64张图片
@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    if request.method == 'POST':
        image = request.files['image'].read() # 读取图片
        print("streamer shape : ", streamer.predict([image]).shape)
        object_name = streamer.predict([image])[0] # 预测输出
        return jsonify({'object_name':object_name})

if __name__ == '__main__':
    app.run(debug=True)
