import gradio as gr
import torch
import soundfile as sf
from speechbrain.inference.separation import SepformerSeparation as separator
import os


# defineing model class
class SepformerFineTune(torch.nn.Module):
    def __init__(self, model):
        super(SepformerFineTune, self).__init__()
        self.model = model
        # disabling gradient computation
        for parms in self.model.parameters():
            parms.requires_grad = False
        
        # enable gradient computation for the last layer
        named_layers = dict(model.named_modules())
        for name, layer in named_layers.items():
            # print(f"Name: {name}, Layer: {layer}")
            if name == "mods.masknet.output.0":
                for param in layer.parameters():
                    param.requires_grad = True
            if name == "mods.masknet.output_gate":
                for param in layer.parameters():
                    param.requires_grad = True
            

        # printing all tranble parameters
        # for model_name, model_params in model.named_parameters():
        #     print(f"Model Layer Name: {model_name}, Model Params: {model_params.requires_grad}")
    def forward(self, mix):
        est_sources = self.model.separate_batch(mix)
        return est_sources[:,:,0], est_sources[:,:,1] # NOTE: Working with 2 sources ONLY


class SourceSeparationApp:
    def __init__(self, model_path,device="cpu"):
        self.model = self.load_model(model_path)
        self.device = device

    def load_model(self, model_path):
        model = separator.from_hparams(source="speechbrain/sepformer-wsj03mix", savedir='pretrained_models/sepformer-wsj03mix', run_opts={"device": device})
        checkpoint = torch.load(model_path)
        fine_tuned_model = SepformerFineTune(model)
        fine_tuned_model.load_state_dict(checkpoint["model"])
        return fine_tuned_model

    def separate_sources(self, audio_file):
        # Load input audio
        # print(f"[LOG] Audio file: {audio_file}")
        input_audio_tensor, sr = audio_file[1], audio_file[0]

        if self.model is None:
            return "Error: Model not loaded."

        # sending input audio to PyTorch tensor
        input_audio_tensor = torch.tensor(input_audio_tensor,dtype=torch.float).unsqueeze(0)
        input_audio_tensor = input_audio_tensor.to(self.device)
        
        # Source separation using the loaded model
        self.model.to(self.device)
        self.model.eval()
        with torch.inference_mode():
            # print(f"[LOG] mix shape: {mix.shape}, s1 shape: {s1.shape}, s2 shape: {s2.shape}, noise shape: {noise.shape}")
            source1,source2 = self.model(input_audio_tensor)


        # Save separated sources
        sf.write("source1.wav", source1.squeeze().cpu().numpy(), sr)
        sf.write("source2.wav", source2.squeeze().cpu().numpy(), sr)

        return "Separation completed", "source1.wav", "source2.wav"

    def run(self):
        audio_input = gr.Audio(label="Upload or record audio")
        output_text = gr.Label(label="Status:") 
        audio_output1 = gr.Audio(label="Source 1", type="filepath",)
        audio_output2 = gr.Audio(label="Source 2", type="filepath",)
        gr.Interface(
            fn=self.separate_sources,
            inputs=audio_input,
            outputs=[output_text, audio_output1, audio_output2],
            title="Audio Source Separation",
            description="Separate sources from a mixed audio signal.",
            examples=[["examples/" + example] for example in os.listdir("examples")],
            allow_flagging=False
        ).launch()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "fine_tuned_sepformer-wsj03mix-7sec.ckpt"  # Replace with your model path
    app = SourceSeparationApp(model_path, device=device)
    app.run()
