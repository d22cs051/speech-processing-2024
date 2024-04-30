import gradio as gr
import torch
from model import Model
from config import Config

import warnings
# warnings.filterwarnings('ignore')

# making config object
config = Config()



def infrence(audio_file1):
    print(f"[LOG] Audio file: {audio_file1}")

class DFSeparationApp:
    def __init__(self, model_path,device="cpu"):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(self.device)


    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        fine_tuned_model = Model(
            args=config,
            device=self.device
        )
        fine_tuned_model.load_state_dict(checkpoint["model"])
        print("[LOG] Model loaded successfully.")
        return fine_tuned_model

    def predict(self, audio_file):
        # Load the audio file
        print(f"[LOG] Audio file: {audio_file}")
        audio_tensor = torch.tensor(audio_file[1][:64600],dtype=torch.float).unsqueeze(0)
        print(f"[LOG] Audio tensor shape: {audio_tensor.shape}")
        with torch.no_grad():
            # Make prediction
            output = self.model(audio_tensor)
            probs = output.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            print(f"[LOG] Prediction: {preds.item()}")
            print(f"[LOG] Probability: {probs.max().item()}")
        pred_str = "Fake" if preds.item() == 1 else "Real"
        return pred_str, probs.max().item()
    
    def run(self):
        print(f"[LOG] Running the app...")
        # gradio interface
        audio_input1 = gr.Audio(label="Upload or record audio")
        prediction = gr.Label(label="Prediction:")
        prob = gr.Label(label="Probability:")
        gr.Interface(
            fn=self.predict,
            inputs=[audio_input1],
            outputs=[prediction, prob],
            title="DF Separation",
            description="This app classify the audio samples into Real and Fake.",
            examples=[
                ["samples/Fake/download (5).wav","1"],
                ["samples/Fake/fake1_1.wav","1"],
                ["samples/Real/Central Avenue 1.wav","0"],
                ["samples/Real/hindi.mp3","0"],
            ]
        ).launch(quiet=False,server_name="0.0.0.0")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOG] Device: {device}")
    model_path = "models/for_trained_model.ckpt"  # Replace with your model path
    app = DFSeparationApp(model_path, device=device)
    app.run()
