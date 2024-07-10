from flask import Flask, request, jsonify
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys
from label_studio_sdk import Client
from datetime import datetime

app = Flask(_name_)

# Label Studio URL ve API anahtarı
LS_URL = "http://localhost:8080"
LS_API_TOKEN = "c01a60e55e05156790ad3e548f813daa151dd67e"
client = Client(url=LS_URL, api_key=LS_API_TOKEN)

# Label Studio için etiketleme konfigürasyonu
label_config = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="iris" background="#FFA39E"/>
  </RectangleLabels>
</View>
"""

class YOLOv8Model(LabelStudioMLBase):
    def _init_(self, **kwargs):
        super()._init_(label_config=label_config, **kwargs)
        self.from_name, self.to_name, self.value, _ = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        self.labels = ['iris']
        self.model = YOLO("best.pt")

    def predict(self, tasks, **kwargs):
        task = tasks[0]
        img_url = task['data'][self.value]
        full_img_url = LS_URL + img_url
        headers = {"Authorization": "Token " + LS_API_TOKEN}

        try:
            img_data = requests.get(full_img_url, headers=headers).content
            img = Image.open(BytesIO(img_data))
        except UnidentifiedImageError as e:
            return {"error": "Could not identify image file."}, 400

        width, height = img.size
        predictions = []
        total_score = 0

        results = self.model.predict(img)
        for res in results:
            for idx, pred in enumerate(res.boxes):
                xyxy = pred.xyxy[0].tolist()
                predictions.append({
                    "id": str(idx),
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "score": pred.conf.item(),
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": xyxy[0] / width * 100,
                        "y": xyxy[1] / height * 100,
                        "width": (xyxy[2] - xyxy[0]) / width * 100,
                        "height": (xyxy[3] - xyxy[1]) / height * 100,
                        "rectanglelabels": [self.labels[int(pred.cls.item())]]
                    }})
                total_score += pred.conf.item()

        final_pred = {
            "result": predictions,
            "score": total_score / (idx + 1),
            "model_version": "v8n"
        }
        
        self.send_preds_to_label_studio(task, final_pred)
        print(final_pred)
        return final_pred
        
    def send_preds_to_label_studio(self, task, final_pred):
        proj_id = task.get('project')
        task_id = task['id']

        if proj_id is None:
            raise ValueError("Project ID is missing in the task data.")

        project = client.get_project(proj_id)
        
        formatted_preds = [{
            'id': task_id,
            'model_version': final_pred['model_version'],
            'created_ago': 'just now',
            'result': final_pred['result'],
            'score': final_pred['score'],
            'cluster': None,
            'neighbors': None,
            'mislabeling': 0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'task': task_id,
            'project': proj_id
        }]

        project.create_predictions(formatted_preds)

    def fit(self, completions, workdir=None, **kwargs):
        import random
        return {'random': random.randint(1, 10)}

model = YOLOv8Model()

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tasks = data['tasks']
    preds = model.predict(tasks)
    return jsonify(preds)

if _name_ == '_main_':
    app.run(debug=False, port=9090)