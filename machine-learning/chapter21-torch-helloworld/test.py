import os

project_path = os.path.dirname(__file__)
data_path = os.path.join(project_path, 'data')
output_path = os.path.join(project_path, 'outputs')
model_pkl_path = os.path.join(output_path, 'model.pkl')
model_onnx_path = os.path.join(output_path, 'model.onnx')

print(project_path)
print(data_path)
print(output_path)
print(model_pkl_path)
print(model_onnx_path)
