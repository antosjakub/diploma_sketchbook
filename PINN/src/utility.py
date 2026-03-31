

import json
def json_dump(file_path, d):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=4)

def json_load(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        d = json.load(f)
    return d


def get_duration(dt):
    h = dt // 3600
    m = (dt - 3600 * h) // 60
    s = dt - 3600 * h - 60 * m
    return int(h),int(m),s


from torch.profiler import profile, ProfilerActivity
from contextlib import nullcontext

class Profiler():
    def __init__(self,report_filename, start_step, end_step):
        self.report_filename = report_filename
        self.start_step = start_step
        self.end_step = end_step

    def make(self) -> None:
        self.prof_ctx = profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )

    def start(self, si):
        if si == self.start_step:
            self.prof_ctx.__enter__()
            print(f"\n[Profiler] Started at step {si+1}")

    def exit(self, si):
        if si == self.end_step-1:
            self.prof_ctx.__exit__(None, None, None)
            print(f"\n[Profiler] Stopped. Results:")
            prof_report = self.prof_ctx.key_averages().table(sort_by="cpu_time_total", row_limit=20)
            print(prof_report)
            # save profiler report
            #prof_ctx.export_chrome_trace(f"run_latest/{profiler_report_filename}.json")
            with open(f"{self.report_filename}.txt", "w") as f:
                f.write(prof_report)

#class Profiler_Dummy()
#    def __init__(self):
#        pass
#    def make(self):
#        pass
#    def start(self, si):
#        pass
#    def exit(self, si):
#        pass


def layers_from_string(layers_string):
    return list(map(lambda x: int(x), layers_string.split(",")))



import inspect
def get_module_classes(module):
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
    }



def header(dir_name):
    import torch
    import architecture
    import pde_models
    print(f"Will be working in directory '{dir_name}'...")

    model_metadata = json_load(f"{dir_name}/model_metadata.json")
    pde_metadata = json_load(f"{dir_name}/pde_metadata.json")

    d = model_metadata["args"]["d"]
    D = d+1
    model_class_name = model_metadata["model_class"]
    print(model_class_name)
    model = get_module_classes(architecture)[model_class_name](D, layers_from_string(model_metadata["args"]["layers"]))
    model.load_state_dict(torch.load(f'{dir_name}/model.pth'))
    model.eval()

    pde_class_name = pde_metadata["pde_class"]
    print(pde_class_name)
    pde_model = get_module_classes(pde_models)[pde_class_name](d)
    pde_model.load_pde_metadata(pde_metadata)
    u_analytic = pde_model.u_analytic

    return model, u_analytic, pde_metadata, model_metadata