import gradio as gr
import torch
from torch import nn
from verification import init_model



# model definition
class WaveLMSpeakerVerifi(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = init_model("wavlm_base_plus")
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, auido1, audio2):
        audio1_emb = self.feature_extractor(auido1)
        audio2_emb = self.feature_extractor(audio2)
        similarity = self.cosine_sim(audio1_emb, audio2_emb)
        similarity = (similarity + 1) / 2 # converting (-1,1) -> (0,1)
        return similarity


class SourceSeparationApp:
    def __init__(self, model_path,device="cpu"):
        self.model = self.load_model(model_path)
        self.device = device

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        fine_tuned_model = WaveLMSpeakerVerifi()
        fine_tuned_model.load_state_dict(checkpoint["model"])
        return fine_tuned_model

    def verify_speaker(self, audio_file1, audio_file2):
        # Load input audio
        # print(f"[LOG] Audio file: {audio_file}")
        input_audio_tensor1, sr1 = audio_file1[1], audio_file1[0]
        input_audio_tensor2, sr2 = audio_file2[1], audio_file2[0]

        if self.model is None:
            return "Error: Model not loaded."

        # sending input audio to PyTorch tensor
        input_audio_tensor1 = torch.tensor(input_audio_tensor1,dtype=torch.float).unsqueeze(0)
        input_audio_tensor1 = input_audio_tensor1.to(self.device)
        input_audio_tensor2 = torch.tensor(input_audio_tensor2,dtype=torch.float).unsqueeze(0)
        input_audio_tensor2 = input_audio_tensor2.to(self.device)
        
        # Source separation using the loaded model
        self.model.to(self.device)
        self.model.eval()
        with torch.inference_mode():
            # print(f"[LOG] mix shape: {mix.shape}, s1 shape: {s1.shape}, s2 shape: {s2.shape}, noise shape: {noise.shape}")
            similarity = self.model(input_audio_tensor1, input_audio_tensor2)

        return similarity.item()

    def run(self):
        audio_input1 = gr.Audio(label="Upload or record audio")
        audio_input2 = gr.Audio(label="Upload or record audio")
        output_text = gr.Label(label="Similarity Score Result:")
        gr.Interface(
            fn=self.verify_speaker,
            inputs=[audio_input1, audio_input2],
            outputs=[output_text],
            title="Speaker Verification",
            description="Speaker Verification using fine-tuned Sepformer model.",
            examples = [
                ["samples/844424933481805-705-m.wav", "samples/844424932691175-645-f.wav","0"],
                ["samples/844424931281875-277-f.wav", "samples/844424930801214-277-f.wav","1"],
            ],
        ).launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "fine-tuning-wavlm-base-plus-checkpoint.ckpt"  # Replace with your model path
    app = SourceSeparationApp(model_path, device=device)
    app.run()
