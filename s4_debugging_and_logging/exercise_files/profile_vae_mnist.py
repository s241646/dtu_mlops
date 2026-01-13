import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from vae_mnist_working import Model, Encoder, Decoder


x_dim = 784
hidden_dim = 400
latent_dim = 20

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

model = Model(encoder=encoder, decoder=decoder)

# Load the trained model
model.load_state_dict(torch.load('vae_model.pth'))
model.eval()  # Set to eval mode for inference profiling
print("Model loaded from vae_model.pth")

# Fix inputs to match VAE input shape (batch_size, 784)
inputs = torch.randn(5, x_dim)


with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/vae_mnist")) as prof:
    # multiple iterations of model
    for i in range(10):
        model(inputs)
        prof.step()

# basic show cpu
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# # show input shapes
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

# profile memory usage - set profile_memory=True in the profiler
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# export results
# prof.export_chrome_trace("trace.json")

# to read the trace, open Chrome and go to chrome://tracing and load the trace.json file

# launch tensorboard to visualize
# tensorboard --logdir=./log

# open browser at http://localhost:6006/#pytorch_profiler