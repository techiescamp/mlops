# emp_attr_model/server.py
import pickle
import os
from kserve import Model, ModelServer
from model_class import EmployeeAttritionModel

class EmployeeAttritionServer(Model):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False

    def load(self):
        model_dir = os.getenv("MODEL_DIR", "/mnt/models")  # KServe mounts storageUri here
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(model_dir, "encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)
        with open(os.path.join(model_dir, "column_names.pkl"), "rb") as f:
            column_names = pickle.load(f)
        with open(os.path.join(model_dir, "categories.pkl"), "rb") as f:
            categories = pickle.load(f)

        self.model = EmployeeAttritionModel(model, scaler, encoder, column_names, categories)
        self.ready = True

    def predict(self, payload, headers=None):
        return self.model.predict(payload)  # Call your custom predict

if __name__ == "__main__":
    model = EmployeeAttritionServer("employee-attrition")
    model.load()
    ModelServer().start([model])

    