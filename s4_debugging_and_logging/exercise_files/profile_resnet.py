import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# model = models.resnet18() # use "./log/resnet18"
model = models.resnet34() # try different model, use "./log/resnet34"
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet34")) as prof:
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